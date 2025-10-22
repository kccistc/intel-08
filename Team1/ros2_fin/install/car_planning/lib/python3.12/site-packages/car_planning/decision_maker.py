#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import Twist
from car_msgs.msg import V2VAlert
import math

class DecisionMaker(Node):
    def __init__(self):
        super().__init__('decision_maker')

        # 파라미터 (사용하지 않지만, 다른 노드와의 호환성을 위해 남겨둘 수 있습니다)
        self.declare_parameter('cruise_speed', 0.6)
        self.declare_parameter('slow_speed', 0.4)
        self.declare_parameter('decision_frequency_hz', 10.0)

        # V2X 경고 메시지 구독
        self.sub_alert = self.create_subscription(V2VAlert, '/v2x/alert_struct', self.alert_cb, 10)
        
        # 차량 제어 명령 발행
        self.pub_cmd = self.create_publisher(Twist, '/vehicle/cmd', 10)

        self.last_alert = None  # 마지막으로 수신한 경고 메시지

        # 주기적인 판단을 위한 타이머 설정
        frequency = self.get_parameter('decision_frequency_hz').get_parameter_value().double_value
        self.timer = self.create_timer(1.0 / frequency, self.make_decision)

        self.get_logger().info(f'DecisionMaker started: Publishing commands at {frequency} Hz.')

    def alert_cb(self, msg: V2VAlert):
        """V2VAlert 콜백, 마지막 경고 메시지를 저장합니다."""
        self.last_alert = msg
        self.get_logger().info(f'Received alert: type={msg.type}')

    def make_decision(self):
        """주기적으로 호출되어 주행 결정을 내리고 명령을 발행합니다."""
        
        cruise_speed = self.get_parameter('cruise_speed').get_parameter_value().double_value
        slow_speed = self.get_parameter('slow_speed').get_parameter_value().double_value
        
        cmd = Twist()
        action = 'CRUISE'
        cmd.linear.x = cruise_speed

        # 유효한 경고 메시지가 있는지 확인
        if self.last_alert:
            # TTL / ts 검사
            ttl = float(self.last_alert.ttl_s) if hasattr(self.last_alert, 'ttl_s') else 0.0
            if ttl > 0.0 and hasattr(self.last_alert, 'ts'):
                now = self.get_clock().now()
                msg_time = Time.from_msg(self.last_alert.ts)
                age_s = (now - msg_time).nanoseconds / 1e9
                if age_s > ttl:
                    self.get_logger().info(f'Expired alert. Resuming cruise speed. (age={age_s:.2f}s > ttl={ttl:.2f}s)')
                    self.last_alert = None  # 만료된 경고는 삭제
                else:
                    # 유효한 경고가 있을 경우 감속
                    action = 'SLOW_DOWN'
                    cmd.linear.x = slow_speed
            else: # TTL 정보가 없는 경고는 수신 시점에만 유효하다고 간주하고 바로 삭제
                 action = 'SLOW_DOWN'
                 cmd.linear.x = slow_speed
                 self.last_alert = None


        self.get_logger().info(f'[{action}] -> cmd: v={cmd.linear.x:.2f} m/s')
        cmd.angular.z = 0.0  # 회전은 고려하지 않음
        self.pub_cmd.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = DecisionMaker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()