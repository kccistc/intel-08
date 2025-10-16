#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import Twist
from car_msgs.msg import V2VAlert  # ver, src, seq, ts, type, severity(str), ...

class DecisionMaker(Node):
    def __init__(self):
        super().__init__('decision_maker')

        # 파라미터
        self.declare_parameter('stop_distance_m', 10.0)
        self.declare_parameter('slow_distance_m', 20.0)
        self.declare_parameter('cruise_speed', 1.0)
        self.declare_parameter('slow_speed', 0.3)
        self.declare_parameter('turn_rate', 0.4)

        self.sub = self.create_subscription(V2VAlert, '/v2x/alert_struct', self.alert_cb, 10)
        self.pub = self.create_publisher(Twist, '/vehicle/cmd', 10)

        self.get_logger().info('DecisionMaker: /v2x/alert_struct → /vehicle/cmd (using ts, severity=str)')

    def alert_cb(self, msg: V2VAlert):
        # TTL / ts 검사 (header 없음, ts 사용)
        ttl = float(msg.ttl_s) if hasattr(msg, 'ttl_s') else 0.0
        if ttl > 0.0 and hasattr(msg, 'ts'):
            now = self.get_clock().now()
            msg_time = Time.from_msg(msg.ts)
            age_s = (now - msg_time).nanoseconds / 1e9
            if age_s > ttl:
                self.get_logger().warn(f'Expired alert ignored (age={age_s:.2f}s > ttl={ttl:.2f}s)')
                return

        stop_dist = self.get_parameter('stop_distance_m').get_parameter_value().double_value
        slow_dist = self.get_parameter('slow_distance_m').get_parameter_value().double_value
        cruise = self.get_parameter('cruise_speed').get_parameter_value().double_value
        slow = self.get_parameter('slow_speed').get_parameter_value().double_value
        turn = self.get_parameter('turn_rate').get_parameter_value().double_value

        sev = (msg.severity or '').lower().strip()
        sug = (msg.suggest or '').lower().strip()
        typ = (msg.type or '').lower().strip()

        cmd = Twist()

        # 규칙
        if typ == 'collision' and msg.distance_m < stop_dist:
            action = 'EMERGENCY_STOP'
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        elif (typ in ('obstacle', 'hazard') or sev in ('medium', 'high')) and msg.distance_m < slow_dist:
            action = 'SLOW_DOWN'
            cmd.linear.x = slow
            cmd.angular.z = 0.0

        elif sug == 'reroute':
            action = 'REROUTE_SLOW'
            cmd.linear.x = slow
            cmd.angular.z = 0.0

        elif sug == 'slow_down':
            action = 'SLOW_DOWN_ADVICE'
            cmd.linear.x = slow
            cmd.angular.z = 0.0

        elif sug == 'stop':
            action = 'STOP_ADVICE'
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        else:
            action = 'CRUISE'
            cmd.linear.x = cruise
            cmd.angular.z = 0.0

        self.pub.publish(cmd)
        self.get_logger().info(
            f'[{action}] type={typ}, sev={sev}, dist={msg.distance_m:.1f}m, suggest={sug} → '
            f'cmd: v={cmd.linear.x:.2f} m/s, yaw_rate={cmd.angular.z:.2f} rad/s'
        )

def main(args=None):
    rclpy.init(args=args)
    node = DecisionMaker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
