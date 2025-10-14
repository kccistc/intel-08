#!/usr/bin/env python3
import json, time, math
from typing import Any, Dict, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from std_msgs.msg import String
from car_msgs.msg import V2VAlert
from builtin_interfaces.msg import Time as RosTime


# ====== 정규화 테이블(소문자 기준) ======
TYPE_ALLOW = {
    "collision":"collision", "fire":"fire", "obstacle":"obstacle",
    "hazard":"hazard", "accident":"collision"
}
SEVERITY_ALLOW = {"low":"low", "medium":"medium", "high":"high"}
SUGGEST_ALLOW  = {"slow_down":"slow_down","stop":"stop","reroute":"reroute","keep":"keep"}


def to_ros_time_from_any(ts_any: Any) -> RosTime:
    """float seconds | {sec,nanosec} | None → builtin_interfaces/Time"""
    t = RosTime()
    try:
        # dict with sec/nanosec
        if isinstance(ts_any, dict) and "sec" in ts_any and "nanosec" in ts_any:
            t.sec = int(ts_any.get("sec") or 0)
            t.nanosec = int(ts_any.get("nanosec") or 0)
            return t
        # float/int/string epoch seconds
        secf = float(ts_any if ts_any is not None else time.time())
        sec = int(math.floor(secf))
        nsec = int((secf - sec) * 1e9)
        t.sec, t.nanosec = sec, nsec
        return t
    except Exception:
        now = time.time()
        sec = int(now)
        t.sec, t.nanosec = sec, int((now - sec) * 1e9)
        return t


def _norm_enum(v: Any, table: Dict[str, str], default: str) -> str:
    if not isinstance(v, str):
        return default
    return table.get(v.strip().lower(), default)


def _as_float(x: Any, default: float) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def _as_int(x: Any, default: int) -> int:
    try:
        if x is None or x == "":
            return default
        return int(x)
    except Exception:
        return default


def _unpack_object(obj: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], float]:
    """
    obj 가 nested(hdr/accident/advice) 또는 flat(ver,src,...) 형태 모두 지원.
    return: (hdr, acc, adv, ttl_s)
    """
    if "hdr" in obj or "accident" in obj or "advice" in obj:
        hdr = dict(obj.get("hdr", {}))
        acc = dict(obj.get("accident", {}))
        adv = dict(obj.get("advice", {}))
        ttl = _as_float(obj.get("ttl_s", 10.0), 10.0)
        return hdr, acc, adv, ttl

    # flat fallback
    hdr = {
        "ver": obj.get("ver"),
        "src": obj.get("src"),
        "seq": obj.get("seq"),
        "ts":  obj.get("ts"),
    }
    acc = {
        "type": obj.get("type"),
        "severity": obj.get("severity"),
        "distance_m": obj.get("distance_m"),
        "road": obj.get("road"),
        "lat": obj.get("lat"),
        "lon": obj.get("lon"),
    }
    adv = {"suggest": obj.get("suggest")}
    ttl = _as_float(obj.get("ttl_s", 10.0), 10.0)
    return hdr, acc, adv, ttl


class V2XAlertBridge(Node):
    def __init__(self):
        super().__init__('v2x_alert_bridge')

        # -------- 파라미터(운영 시 유용) --------
        self.alert_in  = self.declare_parameter('alert_in',  '/v2x/alert').value
        self.alert_out = self.declare_parameter('alert_out', '/v2x/alert_struct').value

        self.drop_expired = self.declare_parameter('drop_expired', True).value
        reliability = self.declare_parameter('reliability', 'reliable').value  # 'reliable'|'besteffort'
        history     = self.declare_parameter('history', 'keeplast').value      # 'keeplast'|'keepall'
        depth       = int(self.declare_parameter('depth', 10).value)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE if reliability == 'reliable' else ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST if history == 'keeplast' else HistoryPolicy.KEEP_ALL,
            depth=depth,
        )

        # 절대 토픽을 쓰더라도 런치에서 remap 가능(상대 리매핑 키 사용)
        self.sub = self.create_subscription(String, self.alert_in, self._on_json, qos)
        self.pub = self.create_publisher(V2VAlert, self.alert_out, qos)

        self.get_logger().info(f'V2XAlertBridge: {self.alert_in} -> {self.alert_out}')

        # 통계(60초마다 요약)
        self.rx = 0
        self.tx = 0
        self.drop_ttl = 0
        self.err_parse = 0
        self.err_schema = 0
        self.create_timer(60.0, self._report)

    def _report(self):
        self.get_logger().info(
            f'[stats] rx={self.rx} tx={self.tx} drop_ttl={self.drop_ttl} '
            f'err_parse={self.err_parse} err_schema={self.err_schema}'
        )

    def _on_json(self, msg: String):
        self.rx += 1
        data = msg.data

        # 1) JSON 파싱
        try:
            obj = json.loads(data)
        except Exception as e:
            self.err_parse += 1
            self.get_logger().warning(f'JSON parse failed: {e}')
            return

        # 2) 스키마 해석(flat/nested 모두 수용) + TTL 필터
        try:
            hdr, acc, adv, ttl = _unpack_object(obj)

            if self.drop_expired:
                ts_any = hdr.get("ts", time.time())
                age = None
                # float/str seconds만 나이에 반영 (dict는 to_ros_time에서 처리)
                if isinstance(ts_any, (int, float, str)):
                    try:
                        age = time.time() - float(ts_any)
                    except Exception:
                        age = None
                if age is not None and age > ttl:
                    self.drop_ttl += 1
                    self.get_logger().debug(
                        f'DROP ttl: seq={hdr.get("seq")} age={age:.2f}s > ttl={ttl}'
                    )
                    return

            # 3) 메시지 구성(결측치 안전)
            m = V2VAlert()
            m.ver = _as_int(hdr.get('ver'), 1)
            m.src = str(hdr.get('src') or 'unknown')
            m.seq = _as_int(hdr.get('seq'), 0)
            m.ts  = to_ros_time_from_any(hdr.get('ts'))

            m.type     = _norm_enum(acc.get('type'), TYPE_ALLOW, 'unknown')
            m.severity = _norm_enum(acc.get('severity'), SEVERITY_ALLOW, 'unknown')

            m.distance_m = _as_float(acc.get('distance_m'), 0.0)
            m.road = str(acc.get('road') or '')
            m.lat  = _as_float(acc.get('lat'), 0.0)
            m.lon  = _as_float(acc.get('lon'), 0.0)

            m.suggest = _norm_enum(adv.get('suggest'), SUGGEST_ALLOW, 'keep')
            m.ttl_s   = _as_float(ttl, 10.0)

            # 4) 발행
            self.pub.publish(m)
            self.tx += 1

        except Exception as e:
            self.err_schema += 1
            self.get_logger().warning(f'build V2VAlert failed: {e}')
            return


def main():
    rclpy.init()
    try:
        node = V2XAlertBridge()
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
