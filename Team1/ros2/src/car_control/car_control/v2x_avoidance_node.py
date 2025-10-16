
from typing import Dict, Tuple, Optional
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import String
from geometry_msgs.msg import Twist

try:
    from car_msgs.msg import V2VAlert  # type: ignore
except Exception:
    V2VAlert = None

class V2XAvoidanceNode(Node):
    def __init__(self):
        super().__init__('v2x_avoidance_node')

        # Params
        self.declare_parameter('near_distance_m', 30.0)
        self.declare_parameter('mid_distance_m', 80.0)
        self.declare_parameter('slow_speed', 0.5)
        self.declare_parameter('evade_speed', 0.8)
        self.declare_parameter('stop_speed', 0.0)
        self.declare_parameter('evade_turn_rate', 0.6)
        self.declare_parameter('cmd_hold_s', 2.0)
        self.declare_parameter('decision_period_s', 0.3)
        self.declare_parameter('default_direction', 'left')
        self.declare_parameter('enable_twist', True)
        self.declare_parameter('ttl_fallback_s', 2.5)
        self.declare_parameter('drop_if_no_ttl', False)
        self.declare_parameter('min_severity_for_stop', 4)
        self.declare_parameter('topic_alert', '/v2x/alert_struct')
        self.declare_parameter('topic_cmd', '/vehicle/cmd')
        self.declare_parameter('topic_cmd_twist', '/vehicle/cmd_twist')

        # Fetch
        gp = self.get_parameter
        self.near_d = gp('near_distance_m').value
        self.mid_d = gp('mid_distance_m').value
        self.slow_v = gp('slow_speed').value
        self.ev_v = gp('evade_speed').value
        self.stop_v = gp('stop_speed').value
        self.turn = gp('evade_turn_rate').value
        self.hold = gp('cmd_hold_s').value
        self.period = gp('decision_period_s').value
        self.bias = gp('default_direction').value
        self.twist_en = gp('enable_twist').value
        self.ttl_fb = gp('ttl_fallback_s').value
        self.drop_no_ttl = gp('drop_if_no_ttl').value
        self.min_sev_stop = gp('min_severity_for_stop').value
        topic_alert = gp('topic_alert').value
        topic_cmd = gp('topic_cmd').value
        topic_cmd_twist = gp('topic_cmd_twist').value

        # QoS
        qos_sub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        qos_pub = QoSProfile(depth=10)

        if V2VAlert is None:
            self.get_logger().warn('car_msgs/V2VAlert not found at import; subscription disabled.')
            self.alert_sub = None
        else:
            self.alert_sub = self.create_subscription(V2VAlert, topic_alert, self.on_alert, qos_sub)

        self.pub_cmd = self.create_publisher(String, topic_cmd, qos_pub)
        self.pub_twist = self.create_publisher(Twist, topic_cmd_twist, qos_pub)

        self.last_cmd = ''
        self.last_cmd_time = 0.0
        self.last_decision_time = 0.0
        self.seen_keys: Dict[Tuple[str, int], float] = {}

        self.get_logger().info(f"V2XAvoidanceNode up. Sub:{topic_alert} Pub:{topic_cmd},{topic_cmd_twist}")

    # ---- utils
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    @staticmethod
    def _to_int_sev(val, default=1):
        """Coerce severity to int. Accepts int/float or strings like 'low','med','high','critical','sev3'."""
        if isinstance(val, (int, float)):
            try:
                return int(val)
            except Exception:
                return default
        if isinstance(val, str):
            s = val.strip().lower()
            table = {
                'low': 1, 'lo': 1, 'l': 1,
                'medium': 2, 'med': 2, 'mid': 2, 'm': 2,
                'high': 4, 'hi': 4, 'h': 4,
                'critical': 5, 'crit': 5, 'c': 5,
            }
            if s in table:
                return table[s]
            # e.g., 'sev3', 's3', 'level-4'
            digits = ''.join(ch for ch in s if ch.isdigit())
            if digits:
                try:
                    return int(digits)
                except Exception:
                    return default
        return default

    def _extract(self, msg) -> Optional[dict]:
        """Return flat dict from V2VAlert; robust to flat or nested schemas, and string severities."""
        # 1) Flat schema 먼저 시도 (지금 들어오는 메시지가 이 형태)
        try:
            return {
                'src': getattr(msg, 'src', ''),
                'seq': int(getattr(msg, 'seq', 0) or 0),
                'ts': getattr(msg, 'ts', None),
                'ttl_s': float(getattr(msg, 'ttl_s', 0.0) or 0.0),
                'accident_type': getattr(msg, 'accident_type', getattr(msg, 'type', '')),
                'severity': self._to_int_sev(getattr(msg, 'severity', 1)),
                'distance_m': float(getattr(msg, 'distance_m', 1e9) or 1e9),
                'road': getattr(msg, 'road', ''),
                'lat': float(getattr(msg, 'lat', 0.0) or 0.0),
                'lon': float(getattr(msg, 'lon', 0.0) or 0.0),
                'advice': getattr(msg, 'advice', getattr(msg, 'advice_suggest', '')) or '',
            }
        except Exception:
            pass

        # 2) Nested(hdr/accident/advice) 백업 경로
        try:
            hdr = getattr(msg, 'hdr')
            accident = getattr(msg, 'accident')
            advice = getattr(msg, 'advice')
            return {
                'src': getattr(hdr, 'src', ''),
                'seq': int(getattr(hdr, 'seq', 0) or 0),
                'ts': getattr(hdr, 'ts', None),
                'ttl_s': float(getattr(msg, 'ttl_s', getattr(hdr, 'ttl_s', 0.0)) or 0.0),
                'accident_type': getattr(accident, 'type', ''),
                'severity': self._to_int_sev(getattr(accident, 'severity', 1)),
                'distance_m': float(getattr(accident, 'distance_m', 1e9) or 1e9),
                'road': getattr(accident, 'road', ''),
                'lat': float(getattr(accident, 'lat', 0.0) or 0.0),
                'lon': float(getattr(accident, 'lon', 0.0) or 0.0),
                'advice': getattr(advice, 'suggest', ''),
            }
        except Exception:
            return None

    def _decide(self, d: dict) -> str:
        dist = d['distance_m']; sev = d['severity']; adv = (d['advice'] or '').lower()
        # Near
        if dist <= self.near_d:
            if adv == 'stop' or sev >= self.min_sev_stop: return 'STOP'
            if adv == 'left': return 'EVADE_LEFT'
            if adv == 'right': return 'EVADE_RIGHT'
            return 'SLOW'
        # Mid
        if dist <= self.mid_d:
            if adv == 'stop' and sev >= (self.min_sev_stop - 1): return 'STOP'
            if adv in ('left','right'): return f"EVADE_{adv.upper()}"
            return 'SLOW'
        # Far
        return 'SLOW' if sev >= self.min_sev_stop else 'PROCEED'

    def _publish_cmd(self, cmd: str):
        now = self._now()
        if cmd == self.last_cmd and (now - self.last_cmd_time) < self.hold:
            return
        self.last_cmd = cmd; self.last_cmd_time = now
        self.pub_cmd.publish(String(data=cmd))
        if self.twist_en:
            tw = Twist()
            if cmd == 'STOP':
                tw.linear.x = self.stop_v
            elif cmd.startswith('EVADE'):
                tw.linear.x = self.ev_v
                tw.angular.z = self.turn if cmd.endswith('LEFT') else (-self.turn if cmd.endswith('RIGHT') else 0.0)
            elif cmd == 'SLOW':
                tw.linear.x = self.slow_v
            else:
                tw.linear.x = self.slow_v
            self.pub_twist.publish(tw)

    def on_alert(self, msg):
        now = self._now()
        if (now - self.last_decision_time) < self.period:
            return
        self.last_decision_time = now

        d = self._extract(msg)
        if d is None: return

        # TTL
        ttl = d.get('ttl_s', 0.0) or self.ttl_fb
        if ttl <= 0.0 and self.drop_no_ttl:
            return

        # dedup
        key = (d.get('src',''), d.get('seq',0))
        if key in self.seen_keys and (now - self.seen_keys[key]) < ttl:
            return
        self.seen_keys[key] = now

        cmd = self._decide(d)
        if cmd == 'EVADE_' and self.bias:
            cmd = f'EVADE_{self.bias.upper()}'
        self._publish_cmd(cmd)

def main():
    rclpy.init()
    node = V2XAvoidanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
