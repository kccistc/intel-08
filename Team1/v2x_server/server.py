#!/usr/bin/env python3
"""
V2X Accident Alert - UDP Multicast Server (PC/RSU side)
- Broadcasts small JSON alerts like: "전방 500m 추돌 사고 발생"
"""
import argparse, json, socket, struct, time, hmac, hashlib, sys
from datetime import datetime

def make_alert(args, seq:int):
    alert = {
        "hdr": {"ver": 1, "src": args.src_id, "seq": seq, "ts": time.time()},
        "accident": {
            "type": args.type,
            "severity": args.severity,
            "distance_m": args.distance_m,
            "road": args.road,
            "lat": args.lat, "lon": args.lon
        },
        "advice": {
            "message": args.message or "",
            "suggest": args.suggest
        },
        "ttl_s": args.ttl_s
    }
    return alert

def sign_if_needed(payload_bytes:bytes, key:str):
    if not key: return None
    return hmac.new(key.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()

def main():
    ap = argparse.ArgumentParser(description="V2X Accident Alert - Multicast Server")
    ap.add_argument("--mcast", default="239.20.20.21")
    ap.add_argument("--port", type=int, default=5520)
    ap.add_argument("--iface", default="")
    ap.add_argument("--hz", type=float, default=2.0)
    ap.add_argument("--repeat", action="store_true")
    ap.add_argument("--src-id", default="rsu_server_01")
    ap.add_argument("--hmac-key", default="")
    ap.add_argument("--ttl-s", type=float, default=10.0)

    # payload fields
    ap.add_argument("--type", default="collision")
    ap.add_argument("--severity", default="high")
    ap.add_argument("--distance-m", type=float, default=500.0)
    ap.add_argument("--road", default="segment_A")
    ap.add_argument("--lat", type=float, default=37.12345)
    ap.add_argument("--lon", type=float, default=127.12345)
    ap.add_argument("--suggest", default="slow_down")
    ap.add_argument("--message", default="")
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)

    if args.iface:
        try:
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(args.iface))
        except OSError as e:
            print(f"[WARN] Failed to set iface: {e}", file=sys.stderr)

    print(f"[INFO] Broadcasting to {args.mcast}:{args.port}")

    seq = 0
    def send_once():
        nonlocal seq
        seq += 1
        alert = make_alert(args, seq)
        raw = json.dumps(alert, separators=(',',':')).encode('utf-8')
        sig = sign_if_needed(raw, args.hmac_key)
        if sig:
            alert["sig"] = {"alg":"hmac-sha256","value":sig}
            raw = json.dumps(alert, separators=(',',':')).encode('utf-8')
        sock.sendto(raw, (args.mcast, args.port))
        print(f"[SEND] {datetime.now().isoformat(timespec='seconds')} seq={seq} {alert['accident']['type']}")

    try:
        if args.repeat:
            period = 1.0/args.hz if args.hz>0 else 0.5
            while True:
                send_once()
                time.sleep(period)
        else:
            for _ in range(3):
                send_once()
                time.sleep(0.25)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped")

if __name__ == "__main__":
    main()
