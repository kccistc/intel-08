#!/usr/bin/env python3
import sys, os, time, math, subprocess, argparse
REC_DIR="rec"; OUT_DIR="archive"
os.makedirs(OUT_DIR, exist_ok=True)

def pick_segments(t0, pre=300, post=300, segdur=10):
    lo = int(math.floor((t0-pre)/segdur)*segdur)
    hi = int(math.ceil((t0+post)/segdur)*segdur)
    files=[]
    for ts in range(lo, hi, segdur):
        p = f"{REC_DIR}/seg_{ts}.mp4"
        if os.path.exists(p): files.append(p)
    return files

def concat(files, out):
    if not files:
        raise SystemExit("no segment files found for the requested window")
    tmp="/tmp/concat.txt"
    with open(tmp,"w") as f:
        for p in files: f.write(f"file '{p}'\n")
    # copy 실패시 재인코딩으로 fallback
    r = subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i",tmp,"-c","copy",out])
    if r.returncode != 0:
        subprocess.check_call(["ffmpeg","-y","-f","concat","-safe","0","-i",tmp,"-c:v","libx264","-crf","23","-preset","veryfast",out])
    print("saved:", out)

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ts", type=float, nargs="?", default=time.time())
    ap.add_argument("--pre", type=int, default=300, help="seconds before ts")
    ap.add_argument("--post", type=int, default=300, help="seconds after ts")
    args = ap.parse_args()
    
    files = pick_segments(args.ts, args.pre, args.post)
    out = f"{OUT_DIR}/accident_{int(args.ts)}.mp4"
    concat(files, out)

