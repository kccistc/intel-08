#!/usr/bin/env python3
"""
V2X Accident Alert - UDP Multicast Client (Vehicle side)
- Listens on a multicast group and prints/logs alerts.
- Optional HMAC verification (--hmac-key)
- Optional ROS2 publishing (--ros2), topic: /v2x/alert (std_msgs/String JSON)

Added:
- Duplicate suppression by (src, seq)
- Rate-limited printing (same content prints at least PRINT_GAP_S apart)
- Link watchdog: warn if no messages for 30s

Updated:
- ROS2 publishing options (--topic, --qos, --depth, --ros2-domain)
- HMAC verification uses canonical JSON (sort_keys) by default (toggle with --no-hmac-sort)
"""

import argparse, json, socket, struct, time, sys, hmac, hashlib, threading, subprocess, os
from datetime import datetime

# ======== Added: dedup / rate limit / watchdog state ========
SEEN = {}                                # key: "src#seq" -> last seen ts
MAX_KEEP = 300                           # keep last 300 seq keys
LAST_PRINT = {"summary": None, "ts": 0.0}
PRINT_GAP_S = 5.0                        # same content prints at most every 5s
LAST_RX_TS = time.time()                 # last successful receive time

def _watch_link():
    """Background watcher: warn if no messages for 30s."""
    global LAST_RX_TS
    while True:
        try:
            if time.time() - LAST_RX_TS > 30:
                print("[WARN] no V2X messages in last 30s (link down?)")
                time.sleep(10)
            time.sleep(1)
        except Exception:
            # never kill the watcher
            pass
# ============================================================

def verify_hmac(raw_bytes: bytes, key: str, sig_hex: str) -> bool:
    calc = hmac.new(key.encode("utf-8"), raw_bytes, hashlib.sha256).hexdigest()
    return hmac.compare_digest(calc, sig_hex)

# ------------------ ROS2 helpers (updated) ------------------
def try_ros2_setup(enable: bool, topic: str, qos_mode: str, depth: int, domain: int):
    if not enable:
        return None, None
    try:
        # Domain 설정 (필요 시)
        if domain is not None:
            os.environ["ROS_DOMAIN_ID"] = str(domain)

        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        from std_msgs.msg import String

        rclpy.init()
        node = Node("v2x_alert_client")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE if qos_mode == 'reliable' else ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=depth
        )
        pub = node.create_publisher(String, topic, qos)
        return node, pub
    except Exception as e:
        print(f"[WARN] ROS2 disabled: {e}", file=sys.stderr)
        return None, None
# ------------------------------------------------------------

def beep_once():
    """Try multiple ways to beep once; return True if started."""
    candidates = [
        ["/usr/bin/play","-nq","synth","0.2","sine","880"],
        ["/usr/bin/speaker-test","-t","sine","-f","880","-l","1"],
    ]
    for cmd in candidates:
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except FileNotFoundError:
            continue
        except Exception:
            continue
    return False

def main():
    global LAST_RX_TS

    ap = argparse.ArgumentParser(description="V2X Accident Alert - Multicast Client")
    ap.add_argument("--mcast", default="239.20.20.20")
    ap.add_argument("--port", type=int, default=5520)
    ap.add_argument("--iface", default="", help="join on specific interface IPv4")
    ap.add_argument("--hmac-key", default="", help="shared secret to verify")
    ap.add_argument("--no-hmac-sort", action="store_true",
                    help="do NOT sort keys when verifying HMAC (default: sort keys)")
    ap.add_argument("--log", default="", help="CSV log path")

    # ===== ROS2 options (NEW) =====
    ap.add_argument("--ros2", action="store_true", help="publish to /v2x/alert (std_msgs/String)")
    ap.add_argument("--topic", default="/v2x/alert", help="ROS2 topic name (default: /v2x/alert)")
    ap.add_argument("--qos", choices=["reliable","besteffort"], default="reliable",
                    help="ROS2 reliability (default: reliable)")
    ap.add_argument("--depth", type=int, default=10, help="ROS2 QoS depth (default: 10)")
    ap.add_argument("--ros2-domain", type=int, default=None,
                    help="Set ROS_DOMAIN_ID for this process (optional)")
    # ===============================

    ap.add_argument("--beep", action="store_true", help="beep on critical events")
    ap.add_argument("--require-sig", action="store_true",
                    help="drop packets without valid HMAC signature")
    args = ap.parse_args()

    # ---- socket setup ----
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(('', args.port))
    except OSError as e:
        print(f"[ERROR] bind failed: {e}", file=sys.stderr)
        sys.exit(1)

    group = socket.inet_aton(args.mcast)
    iface = socket.inet_aton(args.iface) if args.iface else struct.pack('=I', socket.INADDR_ANY)
    mreq = group + iface
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    print(f"[INFO] Listening {args.mcast}:{args.port} iface={args.iface or 'ANY'}")

    # logging
    flog = None
    if args.log:
        flog = open(args.log, 'a', buffering=1)
        if flog.tell() == 0:
            flog.write("recv_ts,seq,src,type,severity,distance_m,road,lat,lon,suggest,ok_hmac\n")

    # ---- ROS2 init (updated) ----
    node, pub = try_ros2_setup(args.ros2, args.topic, args.qos, args.depth, args.ros2_domain)

    # start watchdog
    threading.Thread(target=_watch_link, daemon=True).start()

    # recv loop
    try:
        while True:
            data, addr = sock.recvfrom(65535)
            recv_ts = time.time()
            ok_hmac = ""
            try:
                obj = json.loads(data.decode('utf-8'))

                # Extract signature if present, verify over payload without 'sig'
                raw_wo_sig = data
                verified = False
                if "sig" in obj:
                    sig = obj["sig"].get("value","")
                    clone = dict(obj); clone.pop("sig", None)
                    # Canonical JSON (sorted keys) to avoid key-order mismatch
                    if args.no_hmac_sort:
                        raw_wo_sig = json.dumps(clone, separators=(',',':')).encode('utf-8')
                    else:
                        raw_wo_sig = json.dumps(clone, separators=(',',':'), sort_keys=True).encode('utf-8')
                    if args.hmac_key:
                        verified = verify_hmac(raw_wo_sig, args.hmac_key, sig)
                        ok_hmac = "OK" if verified else "BAD"
                    else:
                        ok_hmac = "UNVERIFIED"
                else:
                    ok_hmac = "NO-SIG"

                # require signature (and valid)
                if args.require_sig and not verified:
                    try:
                        hdr_preview = obj.get("hdr", {})
                        print(f"[DROP] seq={hdr_preview.get('seq')} reason=signature_invalid status={ok_hmac}")
                    except Exception:
                        print("[DROP] reason=signature_invalid", ok_hmac)
                    LAST_RX_TS = recv_ts  # prevent false watchdog
                    continue

                hdr = obj.get("hdr", {})

                # ===== duplicate suppression & rate-limited printing =====
                key = f"{hdr.get('src')}#{hdr.get('seq')}"
                if key in SEEN:
                    LAST_RX_TS = recv_ts
                    continue
                SEEN[key] = recv_ts
                if len(SEEN) > MAX_KEEP:
                    for k,_ in sorted(SEEN.items(), key=lambda x: x[1])[:len(SEEN)-MAX_KEEP]:
                        SEEN.pop(k, None)

                acc_preview = obj.get("accident", {})
                adv_preview = obj.get("advice", {})
                try:
                    dist_i = int(float(acc_preview.get("distance_m", 0)))
                except Exception:
                    dist_i = 0
                summary = f"{acc_preview.get('type')}|{acc_preview.get('severity')}|{dist_i}|{adv_preview.get('suggest')}"
                now = recv_ts
                if summary == LAST_PRINT["summary"] and (now - LAST_PRINT["ts"] < PRINT_GAP_S):
                    LAST_RX_TS = recv_ts
                    continue
                LAST_PRINT["summary"] = summary
                LAST_PRINT["ts"] = now
                # ========================================================

                acc = obj.get("accident", {}); adv = obj.get("advice", {})
                ttl_s = float(obj.get("ttl_s", 10.0) or 10.0)
                try:
                    age = recv_ts - float(hdr.get("ts", recv_ts))
                except Exception:
                    age = 0.0
                if age > ttl_s:
                    print(f"[DROP] seq={hdr.get('seq')} age={age:.1f}s > ttl_s={ttl_s}")
                    LAST_RX_TS = recv_ts
                    continue

                line = (f"[RECV] {datetime.now().isoformat(timespec='seconds')} "
                        f"from={addr[0]} seq={hdr.get('seq')} src={hdr.get('src')} "
                        f"type={acc.get('type')} sev={acc.get('severity')} "
                        f"dist={acc.get('distance_m')}m suggest={adv.get('suggest')} {ok_hmac}")
                print(line)

                if flog:
                    flog.write(f"{recv_ts},{hdr.get('seq')},{hdr.get('src')},{acc.get('type')},{acc.get('severity')},"
                               f"{acc.get('distance_m')},{acc.get('road')},{acc.get('lat')},{acc.get('lon')},"
                               f"{adv.get('suggest')},{ok_hmac}\n")

                if pub:
                    try:
                        from std_msgs.msg import String
                        msg = String()
                        # 그대로 전달 (브리지가 hdr/accident/advice/ttl_s 파싱)
                        msg.data = json.dumps(obj, separators=(',',':'))
                        pub.publish(msg)
                    except Exception as pe:
                        print(f"[WARN] ROS2 publish failed: {pe}", file=sys.stderr)

                if args.beep and (acc.get("type") in ("collision","fire") or acc.get("severity") in ("high","HIGH")):
                    if not beep_once():
                        print("[WARN] beep failed: no audio command available?", file=sys.stderr)

                LAST_RX_TS = recv_ts

            except Exception as e:
                print(f"[WARN] malformed: {e}", file=sys.stderr)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped")
    finally:
        try:
            sock.close()
        except Exception:
            pass
        # ROS2 clean shutdown
        try:
            if node is not None:
                node.destroy_node()
        except Exception:
            pass
        try:
            import rclpy
            if 'rclpy' in sys.modules:
                rclpy.shutdown()
        except Exception:
            pass

if __name__ == "__main__":
    main()
