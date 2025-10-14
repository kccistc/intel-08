
#!/usr/bin/env python3

import time, json, subprocess, os

from pathlib import Path


EVENT_DIR = Path("/opt/v2x/events")

SERVER    = ["/usr/bin/python3","/opt/v2x/server.py","--repeat"]

CLIPPER   = ["/usr/bin/python3","/opt/v2x/make_clip.py"]

INDEXER   = ["/usr/bin/python3","/opt/v2x/db_index.py"]

SEEN=set()


def broadcast(evt):

    cmd = SERVER + [

        "--type", evt.get("type","unknown"),

        "--severity", evt.get("severity","high"),

        "--distance-m", str(evt.get("distance_m",500)),

        "--road", evt.get("road","segment_A"),

        "--lat", str(evt.get("lat",0.0)),

        "--lon", str(evt.get("lon",0.0)),

        "--ttl-s","10.0"

    ]

    if os.environ.get("V2X_KEY"):

        cmd += ["--hmac-key", os.environ["V2X_KEY"]]

    subprocess.Popen(cmd)


def make_clip(ts):

    subprocess.Popen(CLIPPER + [str(ts)])


def index_event(evt, clip_path=""):

    # watcher 생성 형태(evt는 평면) → 인덱서가 기대하는 구조로 래핑

    wrapped = {

        "ts": evt.get("ts", time.time()),

        "hdr": {"src": "watcher"},

        "accident": {

            "type": evt.get("type","unknown"),

            "severity": evt.get("severity",""),

            "lat": evt.get("lat"), "lon": evt.get("lon"),

            "road": evt.get("road"),

            "distance_m": evt.get("distance_m"),

        },

        "clip": clip_path

    }

    p = subprocess.Popen(INDEXER, stdin=subprocess.PIPE)

    p.communicate(json.dumps(wrapped).encode())


def main():

    EVENT_DIR.mkdir(parents=True, exist_ok=True)

    while True:

        for p in EVENT_DIR.glob("*.json"):

            if p in SEEN: continue

            try:

                evt=json.loads(p.read_text())

            except Exception as e:

                print("[WARN] bad json:", p, e); SEEN.add(p); continue


            ts = float(evt.get("ts", time.time()))

            # 1) 방송

            broadcast(evt)

            # 2) 클립 생성

            if evt.get("clip_hint", True):

                make_clip(ts)

                clip_path = f"/var/archive/accident_{int(ts)}.mp4"

            else:

                clip_path = ""

            # 3) DB 인덱스 기록

            index_event(evt, clip_path)

            SEEN.add(p)

        time.sleep(1)


if __name__=="__main__": main()

