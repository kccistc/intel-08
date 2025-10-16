#!/usr/bin/env python3
import json, time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import serial
from threading import Lock

from std_msgs.msg import String as MsgString
from geometry_msgs.msg import Twist

class BtCmdBridge(Node):
    def __init__(self):
        super().__init__('bt_cmd_bridge')

        # ===== 파라미터 =====
        self.declare_parameter('device', '/dev/rfcomm0')
        self.declare_parameter('baud', 115200)

        # 기존: topic(한 개) -> 변경: 타입별 토픽 분리
        self.declare_parameter('twist_topic', '/vehicle/cmd')  # Twist 구독용
        self.declare_parameter('string_topic', '')             # String 구독용 (기본 비활성)

        self.declare_parameter('timeout_s', 0.5)   # 워치독
        self.declare_parameter('rate_hz', 30.0)    # 최대 송신 주기 제한

        self.dev_path   = self.get_parameter('device').get_parameter_value().string_value
        self.baud       = self.get_parameter('baud').get_parameter_value().integer_value
        self.twist_topic  = self.get_parameter('twist_topic').get_parameter_value().string_value
        self.string_topic = self.get_parameter('string_topic').get_parameter_value().string_value
        self.timeout_s  = float(self.get_parameter('timeout_s').value)
        self.rate_hz    = float(self.get_parameter('rate_hz').value)

        # ===== 직렬 포트 상태 =====
        self.ser = None
        self.ser_lock = Lock()
        self.last_send = 0.0
        self.last_cmd_time = 0.0
        self.last_cmd = (0.0, 0.0)  # (v, yaw)

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
        self.timer = self.create_timer(0.01, self.spin_once)

        self.get_logger().info(
            f'BT bridge starting: dev={self.dev_path} baud={self.baud} '
            f'twist_topic="{self.twist_topic}" string_topic="{self.string_topic}"'
        )
        self._open_serial()

    def _open_serial(self):
        try:
            with self.ser_lock:
                if self.ser and self.ser.is_open:
                    return
                self.ser = serial.Serial(self.dev_path, self.baud, timeout=0.02)
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
        if (now - self.last_send) < (1.0 / max(1e-3, self.rate_hz)):
            return  # 레이트 제한
        with self.ser_lock:
            if not self.ser or not self.ser.is_open:
                return
            try:
                self.ser.write((line + '\n').encode('ascii'))
                self.last_send = now
            except Exception as e:
                self.get_logger().warn(f'Write failed: {e}')
                self._close_serial()

    def cb_twist(self, msg: Twist):
        v = float(msg.linear.x)
        yaw = float(msg.angular.z)
        self.last_cmd = (v, yaw)
        self.last_cmd_time = time.time()
        self._send_line(f'CMD V:{v:.3f},Y:{yaw:.3f}')

    def cb_str(self, msg: MsgString):
        # 기대: {"v":0.5,"yaw":0.4} or "V:0.5,Y:0.4"
        v, yaw = 0.0, 0.0
        s = msg.data.strip()
        try:
            if s.startswith('{'):
                d = json.loads(s)
                v = float(d.get('v', 0.0))
                yaw = float(d.get('yaw', 0.0))
            else:
                parts = dict([kv.split(':') for kv in s.replace(' ','').split(',')])
                v = float(parts.get('V', 0.0))
                yaw = float(parts.get('Y', 0.0))
        except Exception as e:
            self.get_logger().warn(f'Parse error: {e}, msg={s}')
        self.last_cmd = (v, yaw)
        self.last_cmd_time = time.time()
        self._send_line(f'CMD V:{v:.3f},Y:{yaw:.3f}')

    def spin_once(self):
        # 포트 재연결 시도
        if not (self.ser and self.ser.is_open):
            self._open_serial()

        # 워치독: timeout_s 동안 명령 없으면 정지 명령 1회 발행
        if self.timeout_s > 1e-6 and (time.time() - self.last_cmd_time) > self.timeout_s:
            if self.last_cmd != (0.0, 0.0):
                self.last_cmd = (0.0, 0.0)
                self._send_line('CMD V:0.000,Y:0.000 #timeout')
                self.get_logger().warn('Watchdog STOP sent (timeout)')
                self.last_cmd_time = time.time()

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
