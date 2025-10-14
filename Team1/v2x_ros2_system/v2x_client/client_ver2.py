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
"""
import argparse, json, socket, struct, time, sys, hmac, hashlib, threading
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

def verify_hmac(raw_wo_sig:bytes, key:str, sig_hex:str)->bool:
    calc = hmac.new(key.encode("utf-8"), raw_wo_sig, hashlib.sha256).hexdigest()
    return hmac.compare_digest(calc, sig_hex)

def try_ros2_setup(enable:bool):
    if not enable:
        return None, None
    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String
        rclpy.init()
        node = Node("v2x_alert_client")
        pub = node.create_publisher(String, "/v2x/alert", 10)
        return node, pub
    except Exception as e:
        print(f"[WARN] ROS2 disabled: {e}", file=sys.stderr)
        return None, None

def main():
    global LAST_RX_TS

    ap = argparse.ArgumentParser(description="V2X Accident Alert - Multicast Client")
    ap.add_argument("--mcast", default="239.20.20.20")
    ap.add_argument("--port", type=int, default=5520)
    ap.add_argument("--iface", default="", help="join on specific interface IPv4")
    ap.add_argument("--hmac-key", default="", help="shared secret to verify")
    ap.add_argument("--log", default="", help="CSV log path")
    ap.add_argument("--ros2", action="store_true", help="publish to /v2x/alert (std_msgs/String)")
    args = ap.parse_args()

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

    node, pub = try_ros2_setup(args.ros2)

    # ======== Added: start watchdog thread ========
    threading.Thread(target=_watch_link, daemon=True).start()
    # ==============================================

    while True:
        data, addr = sock.recvfrom(65535)
        recv_ts = time.time()
        ok_hmac = ""
        try:
            obj = json.loads(data.decode('utf-8'))

            # Extract signature if present, verify over payload without 'sig'
            raw_wo_sig = data
            if "sig" in obj:
                sig = obj["sig"].get("value","")
                clone = dict(obj); clone.pop("sig", None)
                raw_wo_sig = json.dumps(clone, separators=(',',':')).encode('utf-8')
                if args.hmac_key:
                    ok_hmac = "OK" if verify_hmac(raw_wo_sig, args.hmac_key, sig) else "BAD"
                else:
                    ok_hmac = "UNVERIFIED"

            hdr = obj.get("hdr", {})

            # ======== Added: duplicate suppression & rate-limited printing ========
            # 1) suppress exact duplicates by (src, seq)
            key = f"{hdr.get('src')}#{hdr.get('seq')}"
            if key in SEEN:
                continue
            SEEN[key] = recv_ts
            if len(SEEN) > MAX_KEEP:
                # drop oldest
                for k,_ in sorted(SEEN.items(), key=lambda x: x[1])[:len(SEEN)-MAX_KEEP]:
                    SEEN.pop(k, None)

            # 2) rate-limit same content (type|severity|distance|suggest)
            acc_preview = obj.get("accident", {})
            adv_preview = obj.get("advice", {})
            try:
                dist_i = int(float(acc_preview.get("distance_m", 0)))
            except Exception:
                dist_i = 0
            summary = f"{acc_preview.get('type')}|{acc_preview.get('severity')}|{dist_i}|{adv_preview.get('suggest')}"
            now = recv_ts
            if summary == LAST_PRINT["summary"] and (now - LAST_PRINT["ts"] < PRINT_GAP_S):
                # same content again too soon → skip print/log
                # (still update watchdog time below)
                LAST_RX_TS = recv_ts
                continue
            LAST_PRINT["summary"] = summary
            LAST_PRINT["ts"] = now
            # =====================================================================

            acc = obj.get("accident", {}); adv = obj.get("advice", {})
            ttl_s = obj.get("ttl_s", 10.0)
            age = recv_ts - float(hdr.get("ts", recv_ts))
            if age > ttl_s:
                print(f"[DROP] seq={hdr.get('seq')} age={age:.1f}s > ttl_s={ttl_s}")
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
                from std_msgs.msg import String
                msg = String(); msg.data = json.dumps(obj, separators=(',',':'))
                pub.publish(msg)

            # ======== Added: update last receive ts for watchdog ========
            LAST_RX_TS = recv_ts
            # ===========================================================

        except Exception as e:
            print(f"[WARN] malformed: {e}", file=sys.stderr)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Stopped")
