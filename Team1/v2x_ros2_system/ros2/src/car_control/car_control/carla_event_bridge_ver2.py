# /ws/src/car_control/car_control/carla_event_bridge.py
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from car_msgs.msg import V2VAlert

# carla_msgs 있으면 사용, 없으면 Float32 더미 토픽으로 대체
USE_CARLA_MSGS = True
try:
    from carla_msgs.msg import CarlaCollisionEvent  # 포크에 따라 CollisionEvent 등
except Exception:
    USE_CARLA_MSGS = False
    from std_msgs.msg import Float32

class CarlaEventBridge(Node):
    def __init__(self):
        super().__init__('carla_event_bridge')
        self.seq = 0
        self.last_pose = None

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
        now = self.get_clock().now().nanoseconds / 1e9
        alert.hdr.ver = "1.0"
        alert.hdr.src = "carla"
        alert.hdr.seq = self.seq
        alert.hdr.ts  = now
        self.seq += 1

        alert.accident.type = "collision"
        # impact_value(2~5)를 1~5 정수로 클램프
        sev = int(max(1, min(5, round(impact_value))))
        alert.accident.severity = sev

        # 위치 정보(임시): 시뮬 좌표 미사용 → 0들
        alert.accident.distance_m = 0.0
        alert.accident.road = "carla_sim"
        alert.accident.lat = 0.0
        alert.accident.lon = 0.0

        alert.advice.suggest = "stop"
        alert.ttl_s = 1.0

        self.pub_alert.publish(alert)
        self.get_logger().info(f"Published V2VAlert (seq={alert.hdr.seq}, sev={sev}, impact={impact_value:.2f})")

    # 실 CARLA 이벤트 콜백
    def on_collision_carla(self, msg):
        impact = getattr(msg, 'normal_impulse', 3.0)  # 필드명 포크별 상이 가능
        self._publish_alert(float(impact))

    # 더미(Float32) 콜백
    def on_collision_dummy(self, msg: Float32):
        self._publish_alert(float(msg.data))

def main():
    rclpy.init()
    rclpy.spin(CarlaEventBridge())
    rclpy.shutdown()

if __name__ == "__main__":
    main()
