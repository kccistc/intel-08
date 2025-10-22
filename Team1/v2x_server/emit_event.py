
#!/usr/bin/env python3

import json, time, uuid, pathlib, argparse

p = argparse.ArgumentParser()

p.add_argument("--type", default="collision", choices=["collision","fire","rollover","blockage","unknown"])

p.add_argument("--severity", default="high")

p.add_argument("--lat", type=float, default=37.12345)

p.add_argument("--lon", type=float, default=127.12345)

p.add_argument("--road", default="segment_A")

p.add_argument("--distance-m", type=float, default=500)

p.add_argument("--ts", type=float, default=time.time())

p.add_argument("--clip-hint", action="store_true")

args = p.parse_args()


evt = {

  "ts": args.ts,

  "type": args.type,

  "severity": args.severity,

  "lat": args.lat, "lon": args.lon,

  "road": args.road,

  "distance_m": args.distance_m,

  "clip_hint": bool(args.clip_hint),

}
out = pathlib.Path("events") / f"{uuid.uuid4().hex}.json"

out.write_text(json.dumps(evt))

print("wrote", out)

