#!/usr/bin/env python3
from http.server import BaseHTTPRequestHandler, HTTPServer
import json, os, sqlite3, time, glob, urllib.parse, mimetypes, traceback

# 현재 폴더 기준으로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE_DIR, "v2x_index.sqlite3")
EVENTS_DIR = os.path.join(BASE_DIR, "events")
ARCHIVE_DIR = os.path.join(BASE_DIR, "archive")
REC_DIR = os.path.join(BASE_DIR, "rec")

HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>V2X Status</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:20px;background:#0b1220;color:#e8eef7}
  h1{font-size:20px;margin:0 0 18px}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
  .card{background:#141c2f;border:1px solid #263149;border-radius:12px;padding:14px}
  .label{font-size:12px;color:#9bb0cf}
  .val{font-size:24px;font-weight:700;margin-top:4px}
  .mono{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace}
  .ts{font-size:12px;color:#a5b7d4;margin-top:10px}
  button{background:#1e2a44;color:#e8eef7;border:1px solid #2d3b5b;border-radius:8px;padding:8px 12px;cursor:pointer}
  button:hover{background:#243355}
  table{width:100%;border-collapse:collapse;margin-top:16px;background:#0f1728}
  th,td{padding:8px 10px;border-bottom:1px solid #23304a;font-size:14px}
  th{color:#9bb0cf;text-align:left}
  tr:hover{background:#111a2d}
  .badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;border:1px solid #2d3b5b}
  .sev-high{color:#ffb4b4;border-color:#ff6b6b}
  .sev-medium{color:#ffd18b;border-color:#ffcc66}
  .sev-low{color:#9be5a2;border-color:#7ce38b}
</style>
</head>
<body>
  <h1>V2X Status</h1>
  <div class="grid">
    <div class="card"><div class="label">Events (last hour)</div><div id="events" class="val">-</div></div>
    <div class="card"><div class="label">Last Event</div><div id="last_event" class="val mono">-</div></div>
    <div class="card"><div class="label">Clips</div><div id="clips" class="val">-</div></div>
    <div class="card"><div class="label">Last Segment</div><div id="last_seg" class="val mono">-</div></div>
  </div>

  <div class="ts" id="now">-</div>

  <div style="margin-top:14px">
    <button onclick="refresh()">Refresh</button>
    <span style="font-size:12px;color:#9bb0cf">Auto-refresh every 5s</span>
  </div>

  <h1 style="margin-top:24px">Recent Events</h1>
  <div style="margin-bottom:8px">
    <label class="label">Limit: </label>
    <select id="limitSel" onchange="loadEvents()">
      <option>10</option><option selected>20</option><option>50</option><option>100</option>
    </select>
  </div>
  <table>
    <thead>
      <tr><th>Time</th><th>Type</th><th>Severity</th><th>Road</th><th>Distance(m)</th><th>Clip</th></tr>
    </thead>
    <tbody id="evtTbody"><tr><td colspan="6" class="label">Loading…</td></tr></tbody>
  </table>

<script>
async function fetchJSON(url){ const r=await fetch(url,{cache:'no-store'}); if(!r.ok) throw new Error('HTTP '+r.status); return r.json(); }
function fmt(ts){ if(!ts) return '-'; const d=new Date(ts*1000); return d.toLocaleString(); }
function sevClass(s){ s=(s||'').toLowerCase(); if(s==='high') return 'sev-high'; if(s==='medium') return 'sev-medium'; return 'sev-low'; }

async function refresh(){
  try{
    const s=await fetchJSON('/status');
    document.getElementById('events').textContent = s.events_last_hour ?? '-';
    document.getElementById('last_event').textContent = fmt(s.last_event_ts);
    document.getElementById('clips').textContent = s.clips ?? '-';
    document.getElementById('last_seg').textContent = fmt(s.last_segment_ts);
    document.getElementById('now').textContent = 'Updated: '+fmt(s.now);
  }catch(e){ document.getElementById('now').textContent = 'Error: '+e; }
}
async function loadEvents(){
  const limit = document.getElementById('limitSel').value || 20;
  try{
    const arr = await fetchJSON('/events?limit='+encodeURIComponent(limit));
    const tb = document.getElementById('evtTbody'); tb.innerHTML='';
    if(arr.length===0){ tb.innerHTML = '<tr><td colspan="6" class="label">No events</td></tr>'; return; }
    arr.forEach(it=>{
      const tr=document.createElement('tr');
      tr.innerHTML = `
        <td class="mono">${fmt(it.ts)}</td>
        <td>${it.type||'-'}</td>
        <td><span class="badge ${sevClass(it.severity)}">${it.severity||'-'}</span></td>
        <td>${it.road||''}</td>
        <td>${it.distance_m??''}</td>
        <td>${it.clip?('<a href="'+it.clip+'" target="_blank">open</a>'):'-'}</td>
      `;
      tb.appendChild(tr);
    });
  }catch(e){
    document.getElementById('evtTbody').innerHTML = '<tr><td colspan="6" class="label">Error: '+e+'</td></tr>';
  }
}
refresh(); loadEvents(); setInterval(()=>{refresh();loadEvents();}, 5000);
</script>
</body>
</html>
"""

def get_stats():
    now=time.time()
    recent=0; last_ts=0
    if os.path.exists(DB):
        conn=sqlite3.connect(DB); c=conn.cursor()
        r=c.execute("SELECT COUNT(*), MAX(ts) FROM events WHERE ts>=?", (now-3600,))
        recent, last_ts = r.fetchone(); conn.close()
    clips=len(glob.glob(os.path.join(ARCHIVE_DIR, "accident_*.mp4")))
    segs=glob.glob(os.path.join(REC_DIR, "seg_*.mp4"))
    seg_last= int(max(os.path.getmtime(p) for p in segs)) if segs else 0
    return {"now":int(now),"events_last_hour":recent,"last_event_ts":int(last_ts or 0),
            "clips":clips,"last_segment_ts":seg_last}

def get_recent(limit:int=20):
    rows=[]
    if os.path.exists(DB):
        conn=sqlite3.connect(DB); c=conn.cursor()
        rows=c.execute("""SELECT ts,type,severity,road,distance_m,clip
                          FROM events ORDER BY ts DESC LIMIT ?""",(limit,)).fetchall()
        conn.close()
    out=[]
    for r in rows:
        clip=(r[5] or "")
        if clip and os.path.isfile(clip):
            clip="/clips/"+os.path.basename(clip)
        out.append({"ts":int(r[0]),"type":r[1],"severity":r[2] or "",
                    "road":r[3] or "", "distance_m":r[4], "clip":clip})
    return out

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            u = urllib.parse.urlparse(self.path)
            path, q = u.path, urllib.parse.parse_qs(u.query)

            # serve archived clips
            if path.startswith("/clips/"):
                p = os.path.join(ARCHIVE_DIR, path.split("/clips/",1)[1])
                if not os.path.isfile(p): self.send_error(404); return
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(p)[0] or "application/octet-stream")
                self.end_headers()
                with open(p,"rb") as f: self.wfile.write(f.read())
                return

            if path=="/":
                b=HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type","text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(b); return

            if path=="/status":
                s=get_stats(); b=json.dumps(s).encode()
                self.send_response(200)
                self.send_header("Content-Type","application/json")
                self.send_header("Cache-Control","no-store")
                self.end_headers()
                self.wfile.write(b); return

            if path=="/events":
                try:
                    limit=int(q.get("limit",["20"])[0]); limit=max(1, min(500, limit))
                except Exception: limit=20
                b=json.dumps(get_recent(limit)).encode()
                self.send_response(200)
                self.send_header("Content-Type","application/json")
                self.send_header("Cache-Control","no-store")
                self.end_headers()
                self.wfile.write(b); return

            self.send_error(404)
        except Exception:
            msg = traceback.format_exc().encode()
            self.send_response(500)
            self.send_header("Content-Type","text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(msg)

if __name__=="__main__":
    HTTPServer(("0.0.0.0", 8088), H).serve_forever()