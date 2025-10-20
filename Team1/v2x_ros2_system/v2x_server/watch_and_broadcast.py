#!/usr/bin/env python3

import time, json, subprocess, os, shutil, argparse
from pathlib import Path

# 현재 폴더 기준으로 설정
BASE_DIR = Path(__file__).parent.absolute()
EVENT_DIR = BASE_DIR / "events"
SEEN_DIR = EVENT_DIR / "seen"  # 처리된 파일을 옮길 폴더
ARCHIVE_DIR = BASE_DIR / "archive"
REC_DIR = BASE_DIR / "rec"

SERVER    = ["python3", str(BASE_DIR / "server.py")]
CLIPPER   = ["python3", str(BASE_DIR / "make_clip.py")]
INDEXER   = ["python3", str(BASE_DIR / "db_index.py")]

SEEN=set()


def broadcast(evt, v2x_key=None):

    cmd = SERVER + [

        "--type", evt.get("type","unknown"),

        "--severity", evt.get("severity","high"),

        "--distance-m", str(evt.get("distance_m",500)),

        "--road", evt.get("road","segment_A"),

        "--lat", str(evt.get("lat",0.0)),

        "--lon", str(evt.get("lon",0.0)),

        "--ttl-s","10.0"

    ]

    # src_id가 있으면 인자로 추가
    if "src_id" in evt:
        cmd += ["--src-id", evt["src_id"]]
    else:
        cmd += ["--src-id", "testing!"]

    if v2x_key:
        cmd += ["--hmac-key", v2x_key]

    subprocess.Popen(cmd)


def make_clip(ts):

    subprocess.Popen(CLIPPER + [str(ts)])


def index_event(evt, clip_path=""):

    # watcher 생성 형태(evt는 평면) → 인덱서가 기대하는 구조로 래핑
    wrapped = {

        "ts": evt.get("ts", time.time()),

        "hdr": {"src": "watcher"},

        "accident": {

            "type": evt.get("type", "unknown"),

            "severity": evt.get("severity", ""),

            "lat": evt.get("lat"), "lon": evt.get("lon"),

            "road": evt.get("road"),

            "distance_m": evt.get("distance_m"),

        },

        "clip": clip_path

    }

    p = subprocess.Popen(INDEXER, stdin=subprocess.PIPE)
    p.communicate(json.dumps(wrapped).encode())


def main():
    # argparse를 사용하여 커맨드 라인 인수 파싱
    parser = argparse.ArgumentParser(description="Watch for event files and broadcast V2X messages.")
    parser.add_argument("--v2x-key", help="HMAC key for V2X message signing.")
    args = parser.parse_args()

    # 필요한 디렉터리들 생성
    EVENT_DIR.mkdir(parents=True, exist_ok=True)
    SEEN_DIR.mkdir(exist_ok=True)  # seen 폴더 생성
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    REC_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Watching for new events in '{EVENT_DIR}'...")

    while True:
        # events 폴더에 있는 json 파일만 대상으로 함
        for p in EVENT_DIR.glob("*.json"):
            
            print(f"[NEW] Processing new event: {p.name}")
            destination = SEEN_DIR / p.name

            try:
                evt_orig = json.loads(p.read_text())
                evt = {}
                # 새로운 형식의 JSON이면 내부 표준 형식으로 변환
                if "camera_id" in evt_orig:
                    print(f"[INFO] Converting new event format from {p.name}")
                    evt["ts"] = time.mktime(time.strptime(evt_orig["timestamp"].split('.')[0], "%Y-%m-%dT%H:%M:%S"))
                    evt["type"] = evt_orig.get("event_type", "unknown")
                    evt["src_id"] = evt_orig.get("camera_id") # camera_id를 src_id로 저장
                    if "location" in evt_orig:
                        evt["lat"] = evt_orig["location"].get("y", 0.0)
                        evt["lon"] = evt_orig["location"].get("x", 0.0)
                    # broadcast를 위한 기본값 설정
                    evt["severity"] = evt_orig.get("severity", "high")
                    evt["distance_m"] = evt_orig.get("distance_m", 500)
                    evt["road"] = evt_orig.get("road", "segment_A")
                else:
                    # 기존 형식의 JSON
                    evt = evt_orig

            except Exception as e:
                print(f"[WARN] Bad json: {p.name}, moving to seen. Error: {e}")
                shutil.move(str(p), str(destination)) # 오류 파일도 이동시켜 재시도 방지
                continue

            ts = float(evt.get("ts", time.time()))

            # 1) 방송 (변환된 evt 사용, v2x_key 전달)
            broadcast(evt, args.v2x_key)

            # 2) 클립 생성
            clip_path = ""
            if evt.get("clip_hint", True):
                make_clip(ts)
                clip_path = str(ARCHIVE_DIR / f"accident_{int(ts)}.mp4")

            # 3) DB 인덱스 기록 (변환된 evt 사용)
            index_event(evt, clip_path)

            # 4) 처리 완료 후 파일을 seen 폴더로 이동
            shutil.move(str(p), str(destination))
            print(f"[DONE] Moved {p.name} to {SEEN_DIR.name}/")

        time.sleep(1)


if __name__=="__main__": main()

