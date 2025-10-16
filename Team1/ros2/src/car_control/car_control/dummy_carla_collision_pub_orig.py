#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from builtin_interfaces.msg import Time
from carla_msgs.msg import CarlaCollisionEvent  # ros-bridge 쪽과 동일 패키지 필요 시 Dummy Msg로 대체
import random, time

class DummyCollisionPub(Node):
    def __init__(self):
        super().__init__('dummy_carla_collision_pub')
        self.pub = self.create_publisher(CarlaCollisionEvent, '/carla/ego_vehicle/collision', 10)
        self.timer = self.create_timer(5.0, self.publish_dummy)

    def publish_dummy(self):
        msg = CarlaCollisionEvent()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "ego_vehicle"
        msg.other_actor_id = 999
        msg.normal_impulse = random.uniform(2.0, 5.0)
        self.pub.publish(msg)
        self.get_logger().info(f"Published dummy collision (impact={msg.normal_impulse:.2f})")

def main():
    rclpy.init()
    node = DummyCollisionPub()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
