
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import Twist
from car_msgs.msg import V2VAlert

class DecisionMaker(Node):
    def __init__(self):
        super().__init__("decision_maker")

        # Parameters
        self.declare_parameter("stop_distance_m", 10.0)
        self.declare_parameter("slow_distance_m", 20.0)
        self.declare_parameter("cruise_speed", 1.0)
        self.declare_parameter("slow_speed", 0.3)
        self.declare_parameter("turn_rate", 0.4)
        self.declare_parameter("no_input_timeout_s", 0.0)  # 0이면 비활성

        # Pub/Sub
        self.sub = self.create_subscription(
            V2VAlert, "/v2x/alert_struct", self.on_alert, 10
        )
        self.pub = self.create_publisher(Twist, "/vehicle/cmd", 10)

        # Deadman timer
        self._last_rx = self.get_clock().now()
        self._timer = self.create_timer(0.2, self._tick)

    # Convenience getters
    def p(self, name:str)->float:
        return float(self.get_parameter(name).get_parameter_value().double_value)

    def on_alert(self, msg: V2VAlert):
        self._last_rx = self.get_clock().now()
        self._decide_and_publish(msg)

    def _tick(self):
        timeout = self.p("no_input_timeout_s")
        if timeout <= 0:
            return
        if (self.get_clock().now() - self._last_rx) > Duration(seconds=timeout):
            cmd = Twist()  # zero = stop
            self.pub.publish(cmd)
            self.get_logger().warn(f"Deadman stop: no alert for > {timeout:.1f}s")

    def _decide_and_publish(self, alert: V2VAlert):
        stop_dist = self.p("stop_distance_m")
        slow_dist = self.p("slow_distance_m")
        cruise    = self.p("cruise_speed")
        slow      = self.p("slow_speed")
        turn      = self.p("turn_rate")

        typ = (alert.type or "").strip().lower()
        sev = (alert.severity or "").strip().lower()
        sug = (alert.suggest or "").strip().lower()
        d   = float(alert.distance_m)

        cmd = Twist()

        # ===== Priority =====
        # 1) Emergency stop (collision & near)
        if typ == "collision" and d < stop_dist:
            action = "EMERGENCY_STOP"
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        # 2) Explicit stop advice
        elif sug == "stop":
            action = "STOP_ADVICE"
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        # 3) Avoid advice (left/right)
        elif sug in ("avoid_left", "avoid_right"):
            action = "AVOID_LEFT" if sug == "avoid_left" else "AVOID_RIGHT"
            # Choose forward speed for avoidance; can be tuned
            cmd.linear.x = max(0.5, slow)
            cmd.angular.z = (+turn) if sug == "avoid_left" else (-turn)

        # 4) Slow or reroute advice
        elif sug in ("slow_down", "reroute"):
            action = "SLOW_DOWN_ADVICE" if sug == "slow_down" else "REROUTE_SLOW"
            cmd.linear.x = slow
            cmd.angular.z = 0.0

        # 5) Hazard/obstacle near → slow
        elif (typ in ("obstacle", "hazard") or sev in ("medium", "high")) and d < slow_dist:
            action = "SLOW_DOWN"
            cmd.linear.x = slow
            cmd.angular.z = 0.0

        # 6) Default cruise
        else:
            action = "CRUISE"
            cmd.linear.x = cruise
            cmd.angular.z = 0.0

        self.get_logger().info(
            f"[{action}] type={typ}, sev={sev}, dist={d:.1f}m, suggest={sug} "
            f"→ cmd: v={cmd.linear.x:.2f} m/s, yaw_rate={cmd.angular.z:.2f} rad/s"
        )
        self.pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = DecisionMaker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
