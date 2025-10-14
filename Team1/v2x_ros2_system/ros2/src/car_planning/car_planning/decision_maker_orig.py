# decision_maker_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String       # VLM 토픽용
from geometry_msgs.msg import Twist # 제어 명령 토픽용
# from car_msgs.msg import V2CInfo    # V2V 정보 토픽용 (가정)

class DecisionMakerNode(Node):
    def __init__(self):
        super().__init__('decision_maker_node')

        # Subscriber 설정 (각 담당자가 발행할 토픽들)
        self.vlm_sub = self.create_subscription(String, '/vlm/description', self.vlm_callback, 10)
        self.v2x_sub = self.create_subscription(String, '/v2x/alert', self.v2x_callback, 10)

        # Publisher 설정 (motor_controller_node로 보낼 제어 명령)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # 변수 초기화
        self.last_vlm_info = None
        self.last_v2x_alert = None

    def vlm_callback(self, msg):
        self.last_vlm_info = msg
        self.get_logger().info(f'VLM says: "{msg.data}"')
        self.make_decision()

    def v2x_callback(self, msg):
        self.last_v2x_alert = msg
        self.get_logger().info(f'V2X Alert: "{msg.data}"')
        self.make_decision()

    def make_decision(self):
        # 🧠 TO-DO: 핵심 주행 로직 구현
        # 이 함수에서 self.last_lane_info, self.last_vlm_info 등을 종합하여
        # 최종 주행 명령을 결정합니다.

        cmd_msg = Twist()

        # 예시 로직: VLM이 "stop"이라는 단어를 포함하면 정지
        if self.last_vlm_info and "stop" in self.last_vlm_info.data:
            cmd_msg.linear.x = 0.0
            cmd_msg.angular.z = 0.0
            self.get_logger().warn('STOP command from VLM!')
        else:
            # 기본 차선 유지 주행 로직 (lane_info 기반)
            cmd_msg.linear.x = 0.2  # 0.2m/s로 직진
            cmd_msg.angular.z = 0.0 # 회전 없음

        self.cmd_pub.publish(cmd_msg)

def main(args=None):
    rclpy.init(args=args)
    node = DecisionMakerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
