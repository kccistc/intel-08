#!/usr/bin/env python3
import sqlite3, json, sys, os, time

# --- 수정된 부분 ---
# 현재 스크립트의 위치를 기준으로 DB 경로를 절대 경로로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE_DIR, "v2x_index.sqlite3")
# --- 수정 끝 ---


SCHEMA="""
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY,

  ts REAL NOT NULL,

  type TEXT NOT NULL,

  severity TEXT,

  lat REAL, lon REAL, road TEXT,

  distance_m REAL,

  src TEXT,

  clip TEXT

);

CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);

"""


def ensure_db():
    # 이제 DB 변수가 절대 경로이므로 os.path.dirname()이 항상 유효한 경로를 반환합니다.
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn=sqlite3.connect(DB)

    conn.executescript(SCHEMA)

    return conn


def main():

    evt=json.loads(sys.stdin.read())

    if evt.get("type")=="heartbeat" or evt.get("accident","").get("type")=="heartbeat":

        return  # 하트비트는 인덱싱 제외

    conn=ensure_db()

    acc = evt.get("accident", evt)  # evt가 평면이든 스키마형이든 대응

    src = evt.get("hdr",{}).get("src", "")

    ts  = float(evt.get("ts", acc.get("ts", time.time())))

    conn.execute("""INSERT INTO events(ts,type,severity,lat,lon,road,distance_m,src,clip)

                    VALUES (?,?,?,?,?,?,?,?,?)""", (

        ts,

        acc.get("type","unknown"),

        acc.get("severity"),

        acc.get("lat"), acc.get("lon"),

        acc.get("road"),

        acc.get("distance_m"),

        src,

        evt.get("clip","")

    ))

    conn.commit(); conn.close()


if __name__=="__main__": main()

