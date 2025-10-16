#!/usr/bin/env python3
import json, time, math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import serial
from threading import Lock

from std_msgs.msg import String as MsgString
from geometry_msgs.msg import Twist

def _lrc_hex(b: bytes) -> str:
    x = 0
    for v in b:
        x ^= v
    return f"{x:02X}"

class BtCmdBridge(Node):
    def __init__(self):
        super().__init__('bt_cmd_bridge')

        # ===== 파라미터 =====
        # 기본 장치를 GPIO UART 직결용으로 변경
        self.declare_parameter('device', '/dev/serial0')
        self.declare_parameter('baud', 115200)

        # 토픽
        self.declare_parameter('twist_topic', '/vehicle/cmd')  # Twist 구독용
        self.declare_parameter('string_topic', '')             # String 구독용 (기본 비활성)

        # 동작 파라미터
        self.declare_parameter('timeout_s', 0.5)    # 워치독: 새 명령 없을 때 정지
        self.declare_parameter('rate_hz', 30.0)     # 최대 송신 주기 제한
        self.declare_parameter('heartbeat_s', 1.0)  # 변동 없어도 하트비트 간격
        self.declare_parameter('v_max', 1.0)        # 속도 상한 (양/음 대칭)
        self.declare_parameter('y_max', 1.0)        # 조향 상한 (양/음 대칭)
        self.declare_parameter('use_lrc', True)     # LRC 무결성 적용 여부

        # UART 직결 안정화 옵션 (신규)
        self.declare_parameter('write_timeout_s', 1.0)
        self.declare_parameter('rtscts', False)     # HW flow off
        self.declare_parameter('xonxoff', False)    # SW flow off

        self.dev_path     = self.get_parameter('device').get_parameter_value().string_value
        self.baud         = int(self.get_parameter('baud').get_parameter_value().integer_value)
        self.twist_topic  = self.get_parameter('twist_topic').get_parameter_value().string_value
        self.string_topic = self.get_parameter('string_topic').get_parameter_value().string_value
        self.timeout_s    = float(self.get_parameter('timeout_s').value)
        self.rate_hz      = float(self.get_parameter('rate_hz').value)
        self.heartbeat_s  = float(self.get_parameter('heartbeat_s').value)
        self.v_max        = float(self.get_parameter('v_max').value)
        self.y_max        = float(self.get_parameter('y_max').value)
        self.use_lrc      = bool(self.get_parameter('use_lrc').value)
        self.write_timeout_s = float(self.get_parameter('write_timeout_s').value)
        self.rtscts       = bool(self.get_parameter('rtscts').value)
        self.xonxoff      = bool(self.get_parameter('xonxoff').value)

        # ===== 직렬 포트 상태 =====
        self.ser = None
        self.ser_lock = Lock()
        self.last_send = 0.0
        self.last_cmd_time = 0.0
        self.last_cmd = (0.0, 0.0)         # (v, y)
        self.last_sent_cmd = (None, None)  # 마지막으로 전송한 (v, y)

        # ===== QoS =====
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        def is_enabled(topic_name: str) -> bool:
            if not topic_name:
                return False
            t = topic_name.strip().lower()
            return t not in ('__off__', 'off', 'none', 'false', '0')

        self.sub_twist = None
        self.sub_str   = None
        if is_enabled(self.twist_topic):
            self.sub_twist = self.create_subscription(Twist, self.twist_topic, self.cb_twist, qos)
            self.get_logger().info(f'Subscribing Twist on {self.twist_topic}')
        if is_enabled(self.string_topic):
            self.sub_str = self.create_subscription(MsgString, self.string_topic, self.cb_str, qos)
            self.get_logger().info(f'Subscribing String on {self.string_topic}')

        # ===== 주기 작업 =====
        tick = 0.01  # 10ms 틱에서 레이트/하트비트/워치독 모두 관리
        self.timer = self.create_timer(tick, self.spin_once)

        self.get_logger().info(
            f'UART bridge starting: dev={self.dev_path} baud={self.baud} '
            f'twist_topic="{self.twist_topic}" string_topic="{self.string_topic}" '
            f'use_lrc={self.use_lrc} rtscts={self.rtscts} xonxoff={self.xonxoff}'
        )
        self._open_serial()

    # ========= 프레임 빌드/전송 유틸 =========
    def _sanitize_and_clamp(self, v: float, y: float):
        if math.isnan(v) or math.isinf(v): v = 0.0
        if math.isnan(y) or math.isinf(y): y = 0.0
        v = max(-self.v_max, min(self.v_max, v))
        y = max(-self.y_max, min(self.y_max, y))
        return v, y

    def _build_frame(self, v: float, y: float) -> str:
        core = f"CMD V:{v:.3f},Y:{y:.3f}"
        if self.use_lrc:
            lrc = _lrc_hex(core.encode('ascii'))
            return f"{core}*{lrc}"
        else:
            return core

    def _send_cmd(self, v: float, y: float, force: bool = False):
        now = time.time()
        min_period = 1.0 / max(1e-3, self.rate_hz)
        due_by_rate = (now - self.last_send) >= min_period

        v, y = self._sanitize_and_clamp(v, y)
        changed = (v != self.last_sent_cmd[0]) or (y != self.last_sent_cmd[1])
        due_by_heartbeat = (now - self.last_send) >= max(0.1, self.heartbeat_s)

        if not (force or changed or (due_by_rate and due_by_heartbeat)):
            return

        line = self._build_frame(v, y)
        self._send_line(line)
        self.last_sent_cmd = (v, y)

    # ========= 시리얼 =========
    def _open_serial(self):
        try:
            with self.ser_lock:
                if self.ser and self.ser.is_open:
                    return
                # GPIO UART 직결: 플로우컨트롤 OFF 권장
                self.ser = serial.Serial(
                    self.dev_path,
                    self.baud,
                    timeout=0.02,
                    write_timeout=self.write_timeout_s,
                    rtscts=self.rtscts,
                    xonxoff=self.xonxoff
                )
            self.get_logger().info('Serial opened')
        except Exception as e:
            self.get_logger().warn(f'Open serial failed: {e}')

    def _close_serial(self):
        with self.ser_lock:
            try:
                if self.ser:
                    self.ser.close()
            except:
                pass
            self.ser = None

    def _send_line(self, line: str):
        now = time.time()
        with self.ser_lock:
            if not self.ser or not self.ser.is_open:
                return
            try:
                self.ser.write((line + '\n').encode('ascii'))
                self.last_send = now
            except Exception as e:
                self.get_logger().warn(f'Write failed: {e}')
                self._close_serial()

    # ========= 콜백 =========
    def cb_twist(self, msg: Twist):
        v = float(msg.linear.x)
        y = float(msg.angular.z)
        v, y = self._sanitize_and_clamp(v, y)
        self.last_cmd = (v, y)
        self.last_cmd_time = time.time()
        self._send_cmd(v, y)

    def cb_str(self, msg: MsgString):
        v, y = 0.0, 0.0
        s = msg.data.strip()
        try:
            if s.startswith('{'):
                d = json.loads(s)
                v = float(d.get('v', 0.0))
                y = float(d.get('yaw', 0.0))
            else:
                parts = dict([kv.split(':') for kv in s.replace(' ','').split(',')])
                v = float(parts.get('V', 0.0))
                y = float(parts.get('Y', 0.0))
        except Exception as e:
            self.get_logger().warn(f'Parse error: {e}, msg={s}')
        v, y = self._sanitize_and_clamp(v, y)
        self.last_cmd = (v, y)
        self.last_cmd_time = time.time()
        self._send_cmd(v, y)

    # ========= 주기 처리 =========
    def spin_once(self):
        if not (self.ser and self.ser.is_open):
            self._open_serial()

        now = time.time()

        # 워치독
        if self.timeout_s > 1e-6 and (now - self.last_cmd_time) > self.timeout_s:
            if self.last_cmd != (0.0, 0.0):
                self.get_logger().warn('Watchdog: STOP (timeout)')
            self.last_cmd = (0.0, 0.0)
            self._send_cmd(0.0, 0.0, force=True)

        # 하트비트
        if (now - self.last_send) >= max(0.1, self.heartbeat_s):
            self._send_cmd(*self.last_cmd, force=True)

def main():
    rclpy.init()
    node = BtCmdBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
