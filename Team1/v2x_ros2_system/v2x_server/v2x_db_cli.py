#!/usr/bin/env python3
import sqlite3, argparse, time
DB="/opt/v2x/v2x_index.sqlite3"
ap=argparse.ArgumentParser()
ap.add_argument("--recent", type=int, default=10, help="show last N")
ap.add_argument("--since", type=float, default=0.0, help="min timestamp (epoch)")
args=ap.parse_args()
conn=sqlite3.connect(DB); c=conn.cursor()
if args.since>0:
    rows=c.execute("SELECT ts,type,severity,road,distance_m,clip FROM events WHERE ts>=? ORDER BY ts DESC LIMIT ?",
                   (args.since,args.recent)).fetchall()
else:
    rows=c.execute("SELECT ts,type,severity,road,distance_m,clip FROM events ORDER BY ts DESC LIMIT ?",
                   (args.recent,)).fetchall()
for r in rows:
    ts,typ,sev,road,dist,clip=r
    print(time.strftime("%F %T", time.localtime(ts)), typ, sev or "", f"{road or ''}", f"{dist or ''}m", clip or "")
