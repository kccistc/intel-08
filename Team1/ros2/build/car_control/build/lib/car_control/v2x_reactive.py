#!/usr/bin/env python3
import json, time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from car_msgs.msg import V2VAlert

class V2XReactive(Node):
    def __init__(self):
        super().__init__('v2x_reactive')

        # 파라미터
        self.declare_parameter('min_distance_m', 600.0)   # 이 거리 이내면 조치
        self.declare_parameter('high_sev_only', False)    # severity==high만 처리할지
        self.declare_parameter('hold_time_s', 5.0)        # 동일 조치를 최소 유지할 시간
        self.declare_parameter('default_cmd', 'keep')     # 아무 조치 없을 때
        self.declare_parameter('compat_json', False)      # ← 선택: JSON 호환 수신

        self.last_cmd = None
        self.last_cmd_ts = 0.0

        # ★ 구조화 토픽 구독 (권장)
        self.sub_struct = self.create_subscription(
            V2VAlert, '/v2x/alert_struct', self._on_alert_struct, 10
        )

        # ★ 선택: 호환 모드(JSON String)도 같이 받기 원하면 켜기
        if bool(self.get_parameter('compat_json').value):
            self.sub_json = self.create_subscription(
                String, '/v2x/alert', self._on_alert_json, 10
            )
            self.get_logger().info('V2XReactive: listening /v2x/alert_struct (V2VAlert) + /v2x/alert (JSON)')
        else:
            self.get_logger().info('V2XReactive: listening /v2x/alert_struct (V2VAlert)')

        self.pub = self.create_publisher(String, '/vehicle/cmd', 10)

    # ===== 공통 의사결정 로직 =====
    def _decide_and_publish(self, typ:str, sev:str, dist:float, suggest:str):
        now = time.time()
        # 파라미터
        min_distance = float(self.get_parameter('min_distance_m').value)
        high_only    = bool(self.get_parameter('high_sev_only').value)
        hold_time    = float(self.get_parameter('hold_time_s').value)
        default_cmd  = str(self.get_parameter('default_cmd').value)

        # 필터링 룰
        if dist > min_distance:
            cmd = default_cmd
        elif high_only and sev != 'high':
            cmd = default_cmd
        else:
            if suggest in ('stop', 'halt'):
                cmd = 'stop'
            elif suggest in ('slow_down', 'caution'):
                cmd = 'slow_down'
            elif suggest in ('reroute', 'route_around'):
                cmd = 'reroute'
            else:
                cmd = 'slow_down' if typ in ('collision', 'fire') else default_cmd

        # rate limit
        if cmd == self.last_cmd and (now - self.last_cmd_ts) < hold_time:
            return

        out = String()
        out.data = json.dumps({
            'cmd': cmd,
            'reason': {'type': typ, 'severity': sev, 'distance_m': dist, 'suggest': suggest},
            'ts': now
        }, separators=(',',':'))

        self.pub.publish(out)
        self.last_cmd, self.last_cmd_ts = cmd, now
        self.get_logger().info(f'cmd={cmd} (type={typ}, sev={sev}, dist={dist}m, suggest={suggest})')

    # ===== 구조화 토픽 콜백 =====
    def _on_alert_struct(self, m: V2VAlert):
        try:
            typ  = m.type
            sev  = m.severity
            dist = float(m.distance_m)
            suggest = m.suggest or 'keep'
        except Exception as e:
            self.get_logger().warn(f'bad struct: {e}')
            return
        self._decide_and_publish(typ, sev, dist, suggest)

    # ===== (선택) JSON 호환 콜백 =====
    def _on_alert_json(self, msg: String):
        try:
            obj = json.loads(msg.data)
            acc = obj.get('accident', {})
            adv = obj.get('advice', {})
            typ = str(acc.get('type', 'unknown'))
            sev = str(acc.get('severity', 'unknown'))
            dist = float(acc.get('distance_m', 1e9) or 1e9)
            suggest = str(adv.get('suggest', 'keep'))
        except Exception as e:
            self.get_logger().warn(f'bad json: {e}')
            return
        self._decide_and_publish(typ, sev, dist, suggest)

def main():
    rclpy.init()
    node = V2XReactive()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
