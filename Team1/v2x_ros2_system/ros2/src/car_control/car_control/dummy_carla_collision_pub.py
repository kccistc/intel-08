#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import random

class DummyCollisionPub(Node):
    def __init__(self):
        super().__init__('dummy_carla_collision_pub')
        self.pub = self.create_publisher(Float32, '/carla/collision_impact', 10)
        self.timer = self.create_timer(5.0, self.publish_dummy)

    def publish_dummy(self):
        impact = random.uniform(2.0, 5.0)  # 2~5 사이 임팩트
        msg = Float32(data=impact)
        self.pub.publish(msg)
        self.get_logger().info(f'Published dummy collision impact={impact:.2f}')

def main():
    rclpy.init()
    rclpy.spin(DummyCollisionPub())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
