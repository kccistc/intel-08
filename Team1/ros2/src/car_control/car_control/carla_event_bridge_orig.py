#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
from carla_msgs.msg import CarlaCollisionEvent  # 이름이 포크별로 다를 수 있음: CollisionEvent 등
from nav_msgs.msg import Odometry
from car_msgs.msg import V2VAlert  # 너의 패키지 기준으로 import 경로 확인

class CarlaEventBridge(Node):
    def __init__(self):
        super().__init__('carla_event_bridge')
        self.seq = 0
        # CARLA 토픽들 (네 환경에 맞게 이름 확인)
        self.sub_collision = self.create_subscription(
            CarlaCollisionEvent, '/carla/ego_vehicle/collision', self.on_collision, 10)
        self.sub_odom = self.create_subscription(
            Odometry, '/carla/ego_vehicle/odometry', self.on_odom, 10)
        self.pub_alert = self.create_publisher(V2VAlert, '/v2x/alert_struct', 10)

        self.last_pose = None

    def on_odom(self, msg: Odometry):
        self.last_pose = msg

    def on_collision(self, msg: CarlaCollisionEvent):
        # 충돌 이벤트를 우리 스키마로 변환
        alert = V2VAlert()
        now = self.get_clock().now().nanoseconds / 1e9

        # hdr
        alert.hdr.ver = "1.0"
        alert.hdr.src = "carla"
        alert.hdr.seq = self.seq
        alert.hdr.ts  = now
        self.seq += 1

        # accident
        alert.accident.type = "collision"
        # severity 매핑(간단 예시: 충돌 임팩트 크기 기반)
        impact = getattr(msg, 'normal_impulse', 1.0)  # 포크별 필드명 다르면 조정
        alert.accident.severity = int(min(max(impact, 1.0), 5.0))  # 1~5 범위 예시

        # 거리/위치(시뮬 환경이므로 차량 자기 위치)
        if self.last_pose:
            p = self.last_pose.pose.pose.position
            alert.accident.distance_m = 0.0
            alert.accident.road = "carla_sim"
            # 위경도 대신 시뮬 좌표→임시로 lat/lon에 넣지 않거나 (0,0) 처리
            alert.accident.lat = 0.0
            alert.accident.lon = 0.0
        else:
            alert.accident.distance_m = 0.0
            alert.accident.road = "carla_sim"
            alert.accident.lat = 0.0
            alert.accident.lon = 0.0

        # advice: 충돌 시 정지 권고
        alert.advice.suggest = "stop"

        # TTL
        alert.ttl_s = 1.0

        self.pub_alert.publish(alert)
        self.get_logger().info(f"Published V2VAlert from CARLA collision (seq={alert.hdr.seq})")

def main():
    rclpy.init()
    node = CarlaEventBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
