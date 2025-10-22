#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.time import Time as RosTime

from car_msgs.msg import V2VAlert

import json
import socket
import struct
import threading
import hmac
import hashlib
import time
import math
from typing import Any, Dict, Tuple

# ====== 정규화 테이블(소문자 기준) ======
TYPE_ALLOW = {
    "collision":"collision", "fire":"fire", "obstacle":"obstacle",
    "hazard":"hazard", "accident":"collision", "accident_detected":"collision"
}
SEVERITY_ALLOW = {"low":"low", "medium":"medium", "high":"high"}
SUGGEST_ALLOW  = {"slow_down":"slow_down","stop":"stop","reroute":"reroute","keep":"keep"}

# ====== Helper Functions (from both files) ======

def verify_hmac(raw_wo_sig:bytes, key:str, sig_hex:str)->bool:
    """HMAC-SHA256 서명을 검증하는 함수"""
    calc = hmac.new(key.encode("utf-8"), raw_wo_sig, hashlib.sha256).hexdigest()
    return hmac.compare_digest(calc, sig_hex)

def to_ros_time_from_any(ts_any: Any) -> RosTime:
    """float seconds | {sec,nanosec} | None → builtin_interfaces/Time"""
    t = RosTime()
    try:
        if isinstance(ts_any, dict) and "sec" in ts_any and "nanosec" in ts_any:
            t.sec = int(ts_any.get("sec") or 0)
            t.nanosec = int(ts_any.get("nanosec") or 0)
            return t
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
        return float(x) if x is not None and x != "" else default
    except Exception:
        return default

def _as_int(x: Any, default: int) -> int:
    try:
        return int(x) if x is not None and x != "" else default
    except Exception:
        return default

def _unpack_object(obj: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], float]:
    """obj 가 nested 또는 flat 형태 모두 지원."""
    if "hdr" in obj or "accident" in obj or "advice" in obj:
        hdr = dict(obj.get("hdr", {}))
        acc = dict(obj.get("accident", {}))
        adv = dict(obj.get("advice", {}))
        ttl = _as_float(obj.get("ttl_s", 10.0), 10.0)
        return hdr, acc, adv, ttl
    # flat fallback
    hdr = { "ver": obj.get("ver"), "src": obj.get("src"), "seq": obj.get("seq"), "ts":  obj.get("ts") }
    acc = { "type": obj.get("type"), "severity": obj.get("severity"), "distance_m": obj.get("distance_m"), "road": obj.get("road"), "lat": obj.get("lat"), "lon": obj.get("lon") }
    adv = {"suggest": obj.get("suggest")}
    ttl = _as_float(obj.get("ttl_s", 10.0), 10.0)
    return hdr, acc, adv, ttl

# ====== Main Listener Thread ======

def comm_listener_thread(node: "V2CCommNode"):
    """UDP 수신, 검증, 파싱 후 구조화된 ROS2 토픽으로 직접 발행하는 스레드"""
    mcast_group = "239.20.20.21"
    mcast_port = 5520
    
    hmac_key = node.get_parameter('hmac_key').get_parameter_value().string_value
    if hmac_key:
        node.get_logger().info('HMAC signature verification is enabled.')
    else:
        node.get_logger().warn('HMAC key is not set. Signature verification is disabled.')

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(('', mcast_port))
    except OSError as e:
        node.get_logger().error(f"Failed to bind to port {mcast_port}: {e}")
        return

    mreq = struct.pack("4sl", socket.inet_aton(mcast_group), socket.INADDR_ANY)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    
    node.get_logger().info(f'V2C listener thread started, listening on {mcast_group}:{mcast_port}')

    while rclpy.ok():
        try:
            raw_data, addr = sock.recvfrom(2048)
            node.rx += 1
            
            # 1) JSON 파싱
            try:
                obj = json.loads(raw_data.decode('utf-8'))
            except Exception as e:
                node.err_parse += 1
                node.get_logger().warning(f'JSON parse failed: {e}')
                continue

            # 2) HMAC 서명 검증
            if hmac_key and "sig" in obj:
                sig_obj = obj.get("sig")
                # sig가 딕셔너리 형태일 경우 value 키의 값을 사용
                sig = sig_obj.get("value", "") if isinstance(sig_obj, dict) else ""
                
                clone = dict(obj)
                clone.pop("sig", None)
                # 정렬 및 공백 제거하여 서명 생성 시와 동일한 문자열 보장
                raw_wo_sig = json.dumps(clone, sort_keys=True, separators=(',',':')).encode('utf-8')
                
                if not sig or not verify_hmac(raw_wo_sig, hmac_key, sig):
                    node.get_logger().warn(f"HMAC verification failed for message from {addr[0]}. Dropping.")
                    node.err_hmac += 1 # HMAC 오류 카운트 추가
                    continue

            # 3) 스키마 해석(flat/nested 모두 수용) + TTL 필터
            try:
                hdr, acc, adv, ttl = _unpack_object(obj)

                if node.drop_expired:
                    ts_any = hdr.get("ts", time.time())
                    age = None
                    if isinstance(ts_any, (int, float, str)):
                        try: age = time.time() - float(ts_any)
                        except Exception: age = None
                    if age is not None and age > ttl:
                        node.drop_ttl += 1
                        continue

                # 4) 메시지 구성(결측치 안전)
                m = V2VAlert()
                m.ver = _as_int(hdr.get('ver'), 1)
                m.src = str(hdr.get('src') or 'unknown')
                m.seq = _as_int(hdr.get('seq'), 0)
                
                # ts 필드를 RosTime 객체로 변환하여 할당
                ros_time = to_ros_time_from_any(hdr.get('ts'))
                m.ts.sec = ros_time.sec
                m.ts.nanosec = ros_time.nanosec

                m.type     = _norm_enum(acc.get('type'), TYPE_ALLOW, 'unknown')
                m.severity = _norm_enum(acc.get('severity'), SEVERITY_ALLOW, 'unknown')
                m.distance_m = _as_float(acc.get('distance_m'), 0.0)
                m.road = str(acc.get('road') or '')
                m.lat  = _as_float(acc.get('lat'), 0.0)
                m.lon  = _as_float(acc.get('lon'), 0.0)
                m.suggest = _norm_enum(adv.get('suggest'), SUGGEST_ALLOW, 'keep')
                m.ttl_s   = _as_float(ttl, 10.0)

                # 5) 발행
                node.pub.publish(m)
                node.tx += 1

            except Exception as e:
                node.err_schema += 1
                node.get_logger().warning(f'Build V2VAlert failed: {e}')

        except Exception as e:
            if rclpy.ok():
                node.get_logger().error(f"V2C listener socket error: {e}")
    sock.close()

# ====== ROS 2 Node Class ======

class V2CCommNode(Node):
    def __init__(self):
        super().__init__('v2c_comm_node')

        # 파라미터 선언
        self.declare_parameter('hmac_key', '')
        self.alert_out = self.declare_parameter('alert_out', '/v2x/alert_struct').value
        self.drop_expired = self.declare_parameter('drop_expired', True).value
        
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.pub = self.create_publisher(V2VAlert, self.alert_out, qos)
        self.get_logger().info(f'V2CCommNode started. Publishing to {self.alert_out}')

        # 통계
        self.rx = 0
        self.tx = 0
        self.drop_ttl = 0
        self.err_parse = 0
        self.err_schema = 0
        self.err_hmac = 0 # HMAC 오류 카운터 추가
        self.create_timer(60.0, self._report)

        # 리스너 스레드 시작
        self.listener_thread = threading.Thread(target=comm_listener_thread, args=(self,), daemon=True)
        self.listener_thread.start()

    def _report(self):
        self.get_logger().info(
            f'[stats] rx={self.rx} tx={self.tx} drop_ttl={self.drop_ttl} '
            f'err_parse={self.err_parse} err_schema={self.err_schema} err_hmac={self.err_hmac}'
        )

def main(args=None):
    rclpy.init(args=args)
    node = V2CCommNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()