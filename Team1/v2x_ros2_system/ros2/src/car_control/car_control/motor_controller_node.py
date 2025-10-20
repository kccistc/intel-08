# motor_controller_node.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import serial
import time

class MotorControllerNode(Node):
    def __init__(self):
        super().__init__('motor_controller_node')

        # Subscriber 설정
        self.cmd_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)

        # 시리얼 포트 설정
        try:
            # 자신의 STM32 시리얼 포트에 맞게 수정! (예: /dev/ttyACM0)
            self.serial_port = serial.Serial('/dev/rfcomm0', 9600, timeout=1)
            self.get_logger().info('Serial port opened successfully.')
            time.sleep(2) # 포트 안정화를 위한 대기 시간
        except Exception as e:
            self.get_logger().error(f'Failed to open serial port: {e}')
            self.serial_port = None

    def cmd_callback(self, msg):
        if self.serial_port is None:
            return

        # Twist 메시지를 STM32로 보낼 프로토콜로 변환
        # 예시: 선속도(linear.x)는 속력, 각속도(angular.z)는 방향으로 변환

        speed = int(msg.linear.x * 100) # 0.0 ~ 1.0 -> 0 ~ 100
        direction = int(msg.angular.z * 50) + 50 # -1.0 ~ 1.0 -> 0 ~ 100

        # 예시 프로토콜: "S,속력,방향\n"
        command = f"S,{speed},{direction}\n"

        try:
            self.serial_port.write(command.encode())
            self.get_logger().info(f'Sent to STM32: {command.strip()}')
        except Exception as e:
            self.get_logger().error(f'Failed to send serial data: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = MotorControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()