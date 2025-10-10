# motor_controller_node.py (CARLA 없이 테스트하기 위한 임시 "Dummy" 버전)
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
# import carla # CARLA 라이브러리를 사용하지 않음

class MotorControllerNode(Node):
    def __init__(self):
        super().__init__('motor_controller_node')
        
        # CARLA 접속 코드를 모두 제거하고, 노드가 시작되었다는 로그만 남김
        self.get_logger().info('Dummy Motor Controller Node has been started.')
        self.get_logger().info('Subscribing to /cmd_vel topic...')

        # Subscriber: decision_maker가 발행하는 제어 명령을 구독하는 것은 동일
        self.cmd_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_callback,
            10)

    def cmd_callback(self, msg: Twist):
        """
        /cmd_vel 토픽을 받아서 CARLA에 보내는 대신, 터미널에 로그로 출력합니다.
        """
        # 받은 Twist 메시지의 내용을 터미널에 INFO 레벨로 출력
        self.get_logger().info(
            f'Received command: [Linear Velocity: x={msg.linear.x:.2f}], [Angular Velocity: z={msg.angular.z:.2f}]'
        )

def main(args=None):
    rclpy.init(args=args)
    node = MotorControllerNode()
    # CARLA 접속 여부와 상관없이 항상 실행
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()