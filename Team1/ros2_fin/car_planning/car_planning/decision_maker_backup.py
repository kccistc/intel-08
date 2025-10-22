#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry  # Odometry 메시지 임포트
from car_msgs.msg import V2VAlert
import math  # 거리 계산을 위한 math 모듈 임포트

class DecisionMaker(Node):
    def __init__(self):
        super().__init__('decision_maker')

        # 파라미터
        self.declare_parameter('stop_distance_m', 10.0)
        self.declare_parameter('slow_distance_m', 20.0)
        self.declare_parameter('cruise_speed', 1.0)
        self.declare_parameter('slow_speed', 0.3)

        # V2X 경고 메시지 구독
        self.sub_alert = self.create_subscription(V2VAlert, '/v2x/alert_struct', self.alert_cb, 10)
        
        # 차량 Odometry 구독
        #self.sub_odom = self.create_subscription(Odometry, '/carla/ego_vehicle/odometry', self.odom_cb, 10)
        
        # 차량 제어 명령 발행
        self.pub_cmd = self.create_publisher(Twist, '/vehicle/cmd', 10)

        self.current_position = None  # 현재 차량 위치를 저장할 변수
        self.current_speed = 0.0  # 현재 차량 속도를 저장할 변수
        self.get_logger().info('DecisionMaker started: Subscribing to /v2x/alert_struct and /carla/ego_vehicle/odometry')

    """
    def odom_cb(self, msg: Odometry):
        #Odometry 콜백, 현재 위치와 속도를 업데이트합니다.
        self.current_position = msg.pose.pose.position
        self.current_speed = msg.twist.twist.linear.x
    """

    def alert_cb(self, msg: V2VAlert):
        # 현재 위치 정보가 없으면 로직을 실행하지 않음
        """
        if self.current_position is None:
            self.get_logger().info("Waiting for current position from odometry...")
            return
        """

        # TTL / ts 검사
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
        
        typ = (msg.type or '').lower().strip()
        cmd = Twist()

        # 순항 속도를 현재 속도로 설정 (단, 멈춰있을 경우 기본 순항 속도로 출발)
        cruise_speed_to_use = self.current_speed if self.current_speed > 0.1 else cruise

        # V2X 메시지 타입이 'accident_detected'일 경우에만 거리 기반 판단 수행
        if typ == 'accident_detected':
            # 실제 거리 계산 (V2VAlert의 lat, lon을 x, y 좌표로 간주)
            dx = self.current_position.x - msg.lat
            dy = self.current_position.y - msg.lon
            distance = math.sqrt(dx**2 + dy**2)

            # 거리 기반 규칙
            if distance < stop_dist:
                action = 'EMERGENCY_STOP'
                cmd.linear.x = 0.0
            elif distance < slow_dist:
                action = 'SLOW_DOWN'
                cmd.linear.x = slow
            else:
                action = 'CRUISE_SAFE_DISTANCE'
                cmd.linear.x = cruise_speed_to_use
            
            self.get_logger().info(
                f'[{action}] type={typ}, calculated_dist={distance:.1f}m → '
                f'cmd: v={cmd.linear.x:.2f} m/s'
            )
        else:
            # 'accident_detected'가 아니면 순항
            action = 'CRUISE'
            cmd.linear.x = cruise_speed_to_use
            self.get_logger().info(f'[{action}] No accident detected. Cruising.')

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
