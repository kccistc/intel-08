#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
from nav_msgs.msg import Odometry
from car_msgs.msg import V2VAlert

# carla_msgs가 없으면 Float32 더미로 대체
USE_CARLA_MSGS = True
try:
    from carla_msgs.msg import CarlaCollisionEvent  # 포크에 따라 이름이 다를 수 있음
except Exception:
    USE_CARLA_MSGS = False
    from std_msgs.msg import Float32

def impact_to_severity(impact: float) -> str:
    # 임팩트(가짜 기준 0~5)를 severity 문자열로 매핑
    if impact >= 4.0:
        return "high"
    if impact >= 2.5:
        return "medium"
    return "low"

class CarlaEventBridge(Node):
    def __init__(self):
        super().__init__('carla_event_bridge')
        self.seq = 0
        self.last_pose = None

        # 선택: CARLA의 오도메트리(있으면 위치 일부 넣을 수 있음)
        self.sub_odom = self.create_subscription(
            Odometry, '/carla/ego_vehicle/odometry', self.on_odom, 10)

        if USE_CARLA_MSGS:
            self.get_logger().info('Using carla_msgs.CarlaCollisionEvent')
            self.sub_collision = self.create_subscription(
                CarlaCollisionEvent, '/carla/ego_vehicle/collision', self.on_collision_carla, 10)
        else:
            self.get_logger().warn('carla_msgs not found. Using Float32 dummy topic /carla/collision_impact')
            self.sub_collision = self.create_subscription(
                Float32, '/carla/collision_impact', self.on_collision_dummy, 10)

        self.pub_alert = self.create_publisher(V2VAlert, '/v2x/alert_struct', 10)

    def on_odom(self, msg: Odometry):
        self.last_pose = msg

    def _publish_alert(self, impact_value: float):
        alert = V2VAlert()

        # 메타
        alert.ver = 1
        alert.src = "carla"
        alert.seq = self.seq
        self.seq += 1

        now = self.get_clock().now().to_msg()
        # now는 builtin_interfaces/Time
        alert.ts = Time(sec=now.sec, nanosec=now.nanosec)

        # 이벤트 본문 (평면 필드)
        alert.type = "collision"
        alert.severity = impact_to_severity(float(impact_value))
        alert.distance_m = 0.0
        alert.road = "carla_sim"

        # 시뮬 좌표 → 실 lat/lon 없음: 0으로 둠(필요 시 변환 로직 후속 추가)
        alert.lat = 0.0
        alert.lon = 0.0

        alert.suggest = "stop"  # 기본 대응
        alert.ttl_s = 1.0

        self.pub_alert.publish(alert)
        self.get_logger().info(
            f"V2VAlert seq={alert.seq} sev={alert.severity} impact={impact_value:.2f}"
        )

    def on_collision_carla(self, msg):
        impact = float(getattr(msg, 'normal_impulse', 3.0))
        self._publish_alert(impact)

    def on_collision_dummy(self, msg):
        self._publish_alert(float(getattr(msg, 'data', 3.0)))

def main():
    rclpy.init()
    rclpy.spin(CarlaEventBridge())
    rclpy.shutdown()

if __name__ == "__main__":
    main()
