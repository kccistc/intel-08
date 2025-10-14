
#!/usr/bin/env python3

import time, json, subprocess, os

from pathlib import Path

EVENT_DIR = Path("/opt/v2x/events")

BROADCAST = ["/usr/bin/python3","/opt/v2x/server.py","--repeat"]  # 서버 스크립트 경로/옵션

CLIPPER   = ["/usr/bin/python3","/opt/v2x/make_clip.py"]

SEEN=set()


def broadcast(evt):

    cmd = BROADCAST + [

        "--type", evt.get("type","unknown"),

        "--severity", evt.get("severity","high"),

        "--distance-m", str(evt.get("distance_m",500)),

        "--road", evt.get("road","segment_A"),

        "--lat", str(evt.get("lat",0.0)),

        "--lon", str(evt.get("lon",0.0)),

        "--ttl-s","10.0"

    ]

    # HMAC 키를 server.env에 이미 넣어 systemd로 실행 중이면 생략 가능

    subprocess.Popen(cmd)


def make_clip(evt):

    ts = float(evt.get("ts", time.time()))

    subprocess.Popen(CLIPPER + [str(ts)])


def main():

    EVENT_DIR.mkdir(parents=True, exist_ok=True)

    while True:

        for p in EVENT_DIR.glob("*.json"):

            if p in SEEN:

                continue

            try:

                evt=json.loads(p.read_text())

            except Exception as e:

                print("[WARN] bad json:", p, e); 

                SEEN.add(p); continue

            broadcast(evt)

            if evt.get("clip_hint", True):

                make_clip(evt)

            SEEN.add(p)

        time.sleep(1)


if __name__=="__main__":

    main()

