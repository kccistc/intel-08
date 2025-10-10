# decision_maker_node.py
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import Twist
from car_msgs.msg import EmergencyEvent

class DecisionMakerNode(Node):
    def __init__(self):
        super().__init__('decision_maker_node')
        self.is_emergency_stop_active = False; self.stop_start_time = None
        self.v2x_sub = self.create_subscription(EmergencyEvent, '/v2x/emergency_event', self.v2x_event_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.decision_timer = self.create_timer(0.05, self.update_decision)
        self.get_logger().info('Decision Maker Node has been started (v2).')

    def v2x_event_callback(self, msg: EmergencyEvent):
        pos_x = msg.position.x
        pos_y = msg.position.y
        self.get_logger().info(
            f"Received V2X Event from '{msg.vehicle_id}', Type: {msg.msg_type}, "
            f"Confidence: {msg.confidence_score:.2f}, Position: ({pos_x:.2f}, {pos_y:.2f})"
        )
        if not self.is_emergency_stop_active:
            # --- string 비교 대신, 정수형 상수(enum)로 조건을 확인 ---
            if msg.msg_type == EmergencyEvent.MSG_TYPE_EMERGENCY_BRAKE and msg.confidence_score > 0.8:
                self.get_logger().warn(f'EMERGENCY BRAKE triggered by V2X from {msg.vehicle_id}! Stopping for 5 seconds.')
                self.is_emergency_stop_active = True
                self.stop_start_time = self.get_clock().now()

    def update_decision(self):
        cmd_msg = Twist()
        if self.is_emergency_stop_active:
            cmd_msg.linear.x = 0.0; cmd_msg.angular.z = 0.0
            elapsed_time = self.get_clock().now() - self.stop_start_time
            if elapsed_time >= Duration(seconds=5):
                self.get_logger().info('5 seconds passed. Resuming normal driving.')
                self.is_emergency_stop_active = False; self.stop_start_time = None
        else:
            cmd_msg.linear.x = 10.0; cmd_msg.angular.z = 0.0
        self.cmd_pub.publish(cmd_msg)

def main(args=None):
    rclpy.init(args=args); node = DecisionMakerNode(); rclpy.spin(node); node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()