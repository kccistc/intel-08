#!/usr/bin/env python3
import argparse, sys, time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class OnceSub(Node):
    def __init__(self, topic):
        super().__init__('expect_cmd_once')
        self.sub = self.create_subscription(Twist, topic, self.cb, 10)
        self.msg = None
    def cb(self, msg): self.msg = msg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--topic', default='/vehicle/cmd')
    ap.add_argument('--expected-linear-x', type=float, required=True)
    ap.add_argument('--expected-angular-z', type=float, required=True)
    ap.add_argument('--tol', type=float, default=0.15)      # 허용 오차
    ap.add_argument('--timeout', type=float, default=5.0)   # 초
    args = ap.parse_args()

    rclpy.init()
    node = OnceSub(args.topic)

    t0 = time.time()
    ok = False
    while rclpy.ok() and (time.time() - t0) < args.timeout:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.msg is not None:
            lx = node.msg.linear.x
            az = node.msg.angular.z
            dlx = abs(lx - args.expected_linear_x)
            daz = abs(az - args.expected-angular-z) if False else abs(az - args.expected_angular_z)
            # 위 한 줄 Python 변수명 타이포 방지
            if dlx <= args.tol and daz <= args.tol:
                print(f'PASS: v={lx:.2f}≈{args.expected_linear_x:.2f}, yaw={az:.2f}≈{args.expected_angular_z:.2f}')
                ok = True
                break
            else:
                print(f'RECV: v={lx:.2f}, yaw={az:.2f} (expect {args.expected_linear_x:.2f}, {args.expected_angular_z:.2f}, tol={args.tol})')
                # 계속 대기하며 최신 메시지로 갱신
    node.destroy_node(); rclpy.shutdown()
    sys.exit(0 if ok else 2)

if __name__ == '__main__':
    main()
