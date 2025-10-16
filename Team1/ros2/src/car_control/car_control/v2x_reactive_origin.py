#!/usr/bin/env python3
import json, time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from car_msgs.msg import V2VAlert

class V2XReactive(Node):
    def __init__(self):
        super().__init__('v2x_reactive')
        # 파라미터 (필요시 런치에서 오버라이드)
        self.declare_parameter('min_distance_m', 600.0)   # 이 거리 이내면 조치
        self.declare_parameter('high_sev_only', False)    # severity==high만 처리할지
        self.declare_parameter('hold_time_s', 5.0)        # 동일 조치를 최소 유지할 시간
        self.declare_parameter('default_cmd', 'keep')     # 아무 조치 없을 때

        self.last_cmd = None
        self.last_cmd_ts = 0.0

        self.sub = self.create_subscription(String, '/v2x/alert', self._on_alert, 10)
        self.pub = self.create_publisher(String, '/vehicle/cmd', 10)
        self.get_logger().info('V2XReactive started: listen /v2x/alert -> publish /vehicle/cmd')

    def _on_alert(self, msg: String):
        now = time.time()
        try:
            obj = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f'bad json: {e}')
            return

        acc = obj.get('accident', {})
        adv = obj.get('advice', {})
        sev = str(acc.get('severity', 'unknown'))
        dist = float(acc.get('distance_m', 1e9) or 1e9)
        typ  = str(acc.get('type', 'unknown'))
        suggest = str(adv.get('suggest', 'keep'))

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
            # 간단 매핑: suggest/accident 기반
            if suggest in ('stop','halt'):
                cmd = 'stop'
            elif suggest in ('slow_down','caution'):
                cmd = 'slow_down'
            elif suggest in ('reroute','route_around'):
                cmd = 'reroute'
            else:
                # 타입 기반 보수적 기본값
                cmd = 'slow_down' if typ in ('collision','fire') else default_cmd

        # rate limit: 같은 명령을 너무 자주 내보내지 않기
        if cmd == self.last_cmd and (now - self.last_cmd_ts) < hold_time:
            return

        out = String()
        out.data = json.dumps({
            'cmd': cmd,                 # stop | slow_down | reroute | keep
            'reason': {'type': typ, 'severity': sev, 'distance_m': dist, 'suggest': suggest},
            'ts': now
        }, separators=(',',':'))

        self.pub.publish(out)
        self.last_cmd, self.last_cmd_ts = cmd, now
        self.get_logger().info(f'cmd={cmd} (type={typ}, sev={sev}, dist={dist}m, suggest={suggest})')

def main():
    rclpy.init()
    node = V2XReactive()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
