#!/usr/bin/env python3
import json, time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from car_msgs.msg import V2VAlert
from builtin_interfaces.msg import Time as RosTime

def to_ros_time(ts: float) -> RosTime:
    t = RosTime()
    t.sec = int(ts)
    t.nanosec = int((ts - t.sec) * 1e9)
    return t

class V2XAlertBridge(Node):
    def __init__(self):
        super().__init__('v2x_alert_bridge')
        self.sub = self.create_subscription(String, '/v2x/alert', self._on_json, 10)
        self.pub = self.create_publisher(V2VAlert, '/v2x/alert_struct', 10)
        self.get_logger().info('V2XAlertBridge: /v2x/alert -> /v2x/alert_struct')

    def _on_json(self, msg: String):
        try:
            obj = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f'bad json: {e}')
            return
        hdr = obj.get('hdr', {})
        acc = obj.get('accident', {})
        adv = obj.get('advice', {})
        ttl = float(obj.get('ttl_s', 10.0) or 10.0)

        m = V2VAlert()
        m.ver = int(hdr.get('ver', 1))
        m.src = str(hdr.get('src', 'unknown'))
        m.seq = int(hdr.get('seq', 0))
        m.ts  = to_ros_time(float(hdr.get('ts', time.time())))
        m.type = str(acc.get('type', 'unknown'))
        m.severity = str(acc.get('severity', 'unknown'))
        try:
            m.distance_m = float(acc.get('distance_m', 0.0) or 0.0)
        except Exception:
            m.distance_m = 0.0
        m.road = str(acc.get('road', ''))
        try:
            m.lat = float(acc.get('lat', 0.0) or 0.0)
            m.lon = float(acc.get('lon', 0.0) or 0.0)
        except Exception:
            m.lat, m.lon = 0.0, 0.0
        m.suggest = str(adv.get('suggest', 'keep'))
        m.ttl_s = float(ttl)

        self.pub.publish(m)

def main():
    rclpy.init()
    rclpy.spin(V2XAlertBridge())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
