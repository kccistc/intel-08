#!/usr/bin/env python3
"""
squat_service_dual.py  (name-first BLE resolver, single-process service)
- 듀얼 BLE 수신(NANO33_L / NANO33_R) ← 이름 우선 탐색(필요 시 MAC 보조)
- 윈도 특징 계산 + 캘리브(MVC/baseline) (좌/우 독립)
- 멀티태스크 넘파이(.npz/.joblib) 또는 규칙식 추론
- AIF/AI_RMS/AI_iEMG/BI + 단계/문구 산출
- reps_pred_dual.tsv (AI 결과) + imu_tempo.tsv (IMU 템포/rep 등) 동시에 append

실행 예:
  python squat_service_dual.py --user-seq --imu-master L --pair-lag-ms 980 --debug
  python squat_service_dual.py --identity-scale --debug   # 스케일러 우회 테스트
"""
import os, csv, time, json, struct, asyncio, argparse
from collections import deque
from typing import Dict, Optional
import numpy as np
import signal, sys  # ← 추가

from scipy.signal import welch

np.seterr(over='ignore', invalid='ignore')  # 넘파이 경고 억제

try:
    import joblib
except Exception:
    joblib = None

from bleak import BleakScanner, BleakClient
# ---------- graceful stop (SIGINT/SIGTERM) ----------
STOP_EVENT: asyncio.Event | None = None

def _on_signal(sig, frame):
    print(f"[SIGNAL] received {sig}, shutting down...", flush=True)
    try:
        if STOP_EVENT and not STOP_EVENT.is_set():
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop:
                loop.call_soon_threadsafe(STOP_EVENT.set)
    except Exception:
        pass

signal.signal(signal.SIGINT, _on_signal)
signal.signal(signal.SIGTERM, _on_signal)

# ---------------------- BLE / EMG params ----------------------
UUID_SVC = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
UUID_TX  = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

FS   = 500
WIN  = int(0.250*FS)  # 125
HOP  = int(0.125*FS)  # 62
EPS  = 1e-8

# ---------------------- PATHS (absolute) ----------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.join(SCRIPT_DIR, "data", "logs")
os.makedirs(BASE_DIR, exist_ok=True)
OUT_TSV    = os.path.join(BASE_DIR, "reps_pred_dual.tsv")
IMU_TSV    = os.path.join(BASE_DIR, "imu_tempo.tsv")

MODELS_DIR     = os.path.join(SCRIPT_DIR, "models")
MT_SCALER_PATH = os.path.join(MODELS_DIR, "mt_scaler.joblib")
MT_SCALER_NPZ  = os.path.join(MODELS_DIR, "mt_scaler_np.npz")   # 버전 독립 스케일러(있으면 자동 사용)
MT_NPZ_PATH    = os.path.join(MODELS_DIR, "mt_model_numpy.npz")

# per-side fallback (단일측)
FI_SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
FI_SCALER_NPZ  = os.path.join(MODELS_DIR, "scaler_np.npz")
FI_NPZ_PATH    = os.path.join(MODELS_DIR, "model_numpy.npz")

# ---------------------- helpers ----------------------
def r2(x, d=4):
    try:
        v=float(x)
        if np.isnan(v): return 0.0
        return round(v, d)
    except:
        return 0.0

def fatigue_stage(fi: float) -> str:
    if fi <= 0.25: return "A_정상"
    if fi <= 0.40: return "B_주의"
    if fi <= 0.60: return "C_보통"
    if fi <= 0.80: return "D_피로"
    return "E_심한피로"

def bi_stage(bi: float) -> str:
    x = float(max(0.0, min(1.0, bi)))
    if x < 0.10: return "A_매우균형"
    if x < 0.20: return "B_양호"
    if x < 0.30: return "C_보통"
    if x < 0.40: return "D_불균형"
    return "E_심한불균형"

def bi_text(bi: float, dir_score: float, side_names=("왼쪽","오른쪽")) -> str:
    left, right = side_names
    pct = int(round(max(0.0, min(1.0, bi))*100))
    return (f"{left}이 {right}보다 {pct}% 불균형" if dir_score>=0 else
            f"{right}이 {left}보다 {pct}% 불균형")

def cv_norm(v):
    v=float(v)
    if v>2.0: v/=100.0
    return float(np.clip(v,0.0,1.0))

# --- 템포 점수/등급(imu_tempo.tsv용) ---
def tempo_score_from_cv(cv_raw: float) -> int:
    try: v=float(cv_raw)
    except: v=0.0
    if v>2.0: v/=100.0
    v = float(np.clip(v,0.0,1.0))
    return int(round(100*(1.0-v)))   # CV 낮을수록 점수↑

def tempo_level_from_score(s: int) -> str:
    s=int(s)
    if s>=80: return "A_매우안정"
    if s>=60: return "B_안정"
    if s>=40: return "C_보통"
    if s>=20: return "D_불안정"
    return "E_매우불안정"

# --- 수치안정 sigmoid & 유한값 보정 ---
def _safe_sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))  # overflow/underflow 안전

def _finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mask = ~np.isfinite(x)
    if mask.any(): x[mask] = 0.0
    return x

# ---------------------- features ----------------------
def rms(x): x=x.astype(np.float32); return float(np.sqrt(np.mean(x*x)+EPS))
def iemg_mav(x): x=x.astype(np.float32); return float(np.mean(np.abs(x)))
def mdf(x, fs=FS):
    x=x.astype(np.float32); x-=np.mean(x)
    if len(x)<8: return 0.0
    nper = min(len(x), 120); nover = nper // 2
    f,Pxx = welch(x, fs=fs, window="hann", nperseg=nper, noverlap=nover, nfft=512)
    if Pxx.size==0: return 0.0
    c=np.cumsum(Pxx); tot=c[-1]
    if tot<=0: return 0.0
    half=tot*0.5; k=int(np.searchsorted(c,half))
    if k<=0: return float(f[0])
    if k>=len(f): return float(f[-1])
    c0,c1=c[k-1],c[k]; f0,f1=f[k-1],f[k]
    frac=(half-c0)/(c1-c0+1e-12); return float(f0+frac*(f1-f0))
def sample_entropy(x, m=2, r=None):
    x=x.astype(np.float32); N=len(x)
    if N<m+2: return 0.0
    if r is None: r=0.2*np.std(x)+EPS
    def _phi(mm):
        cnt=0
        for i in range(N-mm):
            xi=x[i:i+mm]
            for j in range(i+1,N-mm+1):
                if np.max(np.abs(xi-x[j:j+mm]))<=r: cnt+=1
        return cnt
    A=_phi(m+1); B=_phi(m)
    if B==0 or A==0: return 0.0
    return float(-np.log((A+EPS)/(B+EPS)))
def coarse_grain(x, tau):
    if tau<=1: return x
    L=(len(x)//tau)*tau
    if L==0: return x[:0]
    return x[:L].reshape(-1,tau).mean(axis=1)
def msesen(x, taus=(1,2,3,4), m=2):
    vals=[]
    for t in taus:
        cg=coarse_grain(x,t)
        if len(cg)<m+2: vals.append(0.0)
        else: vals.append(sample_entropy(cg, m=m, r=0.2*np.std(cg)+EPS))
    return float(np.mean(vals) if vals else 0.0)

# ---------------------- calibrator ----------------------
class Calibrator:
    def __init__(self):
        self.state="WARMUP"; self.t0=None
        self.active_rms=[]; self.all_rms=[]
        self.rest={"mdf":[], "sampen":[], "msesen":[], "iemg":[]}
        self.MVC=1.0; self.baseline={k:1.0 for k in self.rest}
    def start(self):
        if self.t0 is None: self.t0=time.monotonic()
    def update(self):
        if self.t0 is None: return
        dt=time.monotonic()-self.t0
        if self.state=="WARMUP" and dt>=0.5: self.state="CALIB"
        elif self.state=="CALIB" and dt>=3.5:
            base = self.active_rms or self.all_rms or [1.0]
            self.MVC = float(np.percentile(base, 95))
            if self.MVC < 1e-6: self.MVC=1.0
            #self.MVC = float(np.clip(self.MVC, 50.0, 600.0))   # ★ 상·하한 설정
            for k,arr in self.rest.items():
                self.baseline[k]=float(np.mean(arr)) if arr else 1.0
            self.state="RUN"
            print(f"[{time.strftime('%H:%M:%S')}] CALIB→RUN | MVC={self.MVC:.2f}")
    def feed(self, feats, imu_vel=None):
        self.all_rms.append(feats["rms"]); self.start(); self.update()
        if self.state!="CALIB": return
        if imu_vel is not None and abs(imu_vel)>15.0: self.active_rms.append(feats["rms"])
        else:
            self.rest["mdf"].append(feats["mdf"])
            self.rest["sampen"].append(feats["sampen"])
            self.rest["msesen"].append(feats["msesen"])
            self.rest["iemg"].append(feats["iemg"])
    def normalize(self, feats):
        r_norm=feats["rms"]/max(self.MVC,1e-6)
        def rel(v,b): b=float(b); return 0.0 if abs(b)<1e-6 else float((v-b)/b)
        return {
            "rms_norm":r_norm,
            "dmdf":rel(feats["mdf"], self.baseline["mdf"]),
            "dsampen":rel(feats["sampen"], self.baseline["sampen"]),
            "dmsesen":rel(feats["msesen"], self.baseline["msesen"]),
            "diemg":rel(feats["iemg"], self.baseline["iemg"]),
        }

# ---------------------- per-side engine ----------------------
class SideEngine:
    def __init__(self, name:str):
        self.name=name
        self.raw=deque(maxlen=FS*10)
        self.cal=Calibrator()
        self.imu={
            "ts_ms":0, "pitch_deg":0.0, "pitch_vel_dps":0.0,
            "state":0, "rep_id":0, "desc_ms":0, "rise_ms":0, "tempo_cv":0.0
        }
        self.frame_id=0
        self.latest=None
    def feed_emg(self, samples):
        if self.cal.t0 is None: self.cal.start()
        self.raw.extend(samples)
    def feed_imu(self, ts_ms, pitch_deg, pitch_vel, state, rep_id, desc_ms, rise_ms, tempo_cv):
        self.imu.update({
            "ts_ms":int(ts_ms), "pitch_deg":float(pitch_deg), "pitch_vel_dps":float(pitch_vel),
            "state":int(state), "rep_id":int(rep_id),
            "desc_ms":int(desc_ms), "rise_ms":int(rise_ms), "tempo_cv":float(tempo_cv)
        })
    def process(self):
        if len(self.raw) < WIN: return None
        x=np.array(list(self.raw)[-WIN:], dtype=np.int16).astype(np.float32)
        x-=np.mean(x)
        feats={"rms":rms(x),"mdf":mdf(x),"sampen":sample_entropy(x),"msesen":msesen(x),"iemg":iemg_mav(x)}
        self.cal.feed(feats, imu_vel=self.imu["pitch_vel_dps"])
        if self.cal.state!="RUN": return None
        norm=self.cal.normalize(feats)
        self.frame_id += 1
        out={ "frame_id": self.frame_id,
              "ts_unix": time.time(),
              "rep_id": self.imu["rep_id"],
              "tempo_cv": self.imu["tempo_cv"],
              **norm }
        self.latest = out
        return out

# ---------------------- NumPy models & scalers ----------------------
class DummyScaler:
    def transform(self, X): return X

class NumpyScaler:
    """버전 독립 스케일러(npz: mean/scale만 사용)"""
    def __init__(self, mean, scale):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        self.scale[self.scale == 0] = 1.0
    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        return (X - self.mean) / self.scale

_scaler_cache: dict[str, object] = {}

def load_scaler(path_joblib: Optional[str], path_npz: Optional[str] = None):
    """joblib 우선, 실패 시 npz, 둘 다 없으면 DummyScaler"""
    key = f"{path_joblib}|{path_npz}"
    if key in _scaler_cache:
        return _scaler_cache[key]

    # 1) joblib
    if path_joblib and os.path.exists(path_joblib) and joblib:
        try:
            sc = joblib.load(path_joblib)
            _scaler_cache[key] = sc
            return sc
        except Exception as e:
            print("[WARN] scaler load fail:", e)

    # 2) npz (버전 독립)
    if path_npz and os.path.exists(path_npz):
        try:
            d = np.load(path_npz)
            sc = NumpyScaler(d["mean"], d["scale"])
            _scaler_cache[key] = sc
            print("[INFO] using numpy scaler:", os.path.basename(path_npz))
            return sc
        except Exception as e:
            print("[WARN] numpy scaler load fail:", e)

    # 3) Dummy
    print("[WARN] scaler missing → identity")
    sc = DummyScaler()
    _scaler_cache[key] = sc
    return sc

class MTModelNP:
    def __init__(self, d:Dict[str,np.ndarray]):
        self.Wb1=d.get("Wb1"); self.bb1=d.get("bb1")
        self.Wb2=d.get("Wb2"); self.bb2=d.get("bb2")
        self.W_fi=d.get("W_fi"); self.b_fi=d.get("b_fi")
        self.W_bi=d.get("W_bi"); self.b_bi=d.get("b_bi")
    @staticmethod
    def relu(x): return np.maximum(x,0.0)
    sigm = staticmethod(_safe_sigmoid)
    def ok(self):
        return all([x is not None for x in [self.Wb1,self.bb1,self.Wb2,self.bb2,self.W_fi,self.b_fi,self.W_bi,self.b_bi]])
    def __call__(self, x:np.ndarray):
        x=_finite(x)
        h=self.relu(x@self.Wb1 + self.bb1)
        h=self.relu(h@self.Wb2 + self.bb2)
        fi=self.sigm(h@self.W_fi + self.b_fi)  # (2,)
        bi=self.sigm(h@self.W_bi + self.b_bi)  # (1,)
        return float(fi[0]), float(fi[1]), float(bi[0])

class FIModelNP:
    def __init__(self, d:Dict[str,np.ndarray]):
        self.W1=d.get("W1"); self.b1=d.get("b1")
        self.W2=d.get("W2"); self.b2=d.get("b2")
    @staticmethod
    def relu(x): return np.maximum(x,0.0)
    sigm = staticmethod(_safe_sigmoid)
    def ok(self): return self.W1 is not None and self.b1 is not None and self.W2 is not None and self.b2 is not None
    def __call__(self, x):
        x=_finite(x)
        h=self.relu(x@self.W1 + self.b1)
        y=self.sigm(h@self.W2 + self.b2)
        return float(np.clip(y,0.0,1.0))

def load_mt_model():
    if os.path.exists(MT_NPZ_PATH):
        d=np.load(MT_NPZ_PATH, allow_pickle=True)
        m=MTModelNP({k:d[k] for k in d.files})
        if m.ok():
            print("[MODEL] multitask loaded")
            return m
    print("[MODEL] multitask not found; try per-side")
    return None

def load_fi_model():
    if os.path.exists(FI_NPZ_PATH):
        d=np.load(FI_NPZ_PATH, allow_pickle=True)
        m=FIModelNP({k:d[k] for k in d.files})
        if m.ok():
            print("[MODEL] per-side FI loaded")
            return m
    print("[MODEL] per-side FI not found; using rule")
    return None

def rule_fi_from_norm(dms, dse, dmd, rns, tcv):
    fi_emg = 0.35*dms + 0.30*dse + 0.25*dmd + 0.10*(1.0 - float(np.clip(rns,0.0,1.0)))
    return float(np.clip(0.8*fi_emg + 0.2*tcv, 0.0, 1.0))

def build_mt_features(pl, pr, tempo_cv_master):
    diff_rms  = abs(pl["rms_norm"] - pr["rms_norm"])
    diff_iEMG = abs(pl["diemg"]    - pr["diemg"])
    diff_dmdf = abs(pl["dmdf"]     - pr["dmdf"])
    diff_dse  = abs(pl["dsampen"]  - pr["dsampen"])
    diff_dms  = abs(pl["dmsesen"]  - pr["dmsesen"])
    ratio_rms = (pl["rms_norm"])/(pr["rms_norm"]+1e-6)
    ratio_ie  = (pl["diemg"])/(pr["diemg"]+1e-6)
    xm = np.array([
        pl["rms_norm"], pl["diemg"], pl["dmdf"], pl["dsampen"], pl["dmsesen"],
        pr["rms_norm"], pr["diemg"], pr["dmdf"], pr["dsampen"], pr["dmsesen"],
        diff_rms, diff_iEMG, diff_dmdf, diff_dse, diff_dms,
        ratio_rms, ratio_ie,
        cv_norm(tempo_cv_master)
    ], dtype=np.float32)
    return xm

def align_to_model_dim(xm: np.ndarray, model_in_dim: int) -> np.ndarray:
    if xm.shape[0] == model_in_dim:
        return xm
    if xm.shape[0] > model_in_dim:
        return xm[:model_in_dim]
    out = np.zeros((model_in_dim,), dtype=np.float32)
    out[:xm.shape[0]] = xm
    return out

# ---------------------- BLE resolver (name-first) ----------------------
async def resolve_device(target_name: str,
                         target_addr: Optional[str] = None,
                         tries: int = 8,
                         scan_seconds: float = 4.0):
    """
    이름 우선으로 스캔. (주소가 주어지면 이름 또는 주소 일치 시 채택)
    backoff로 재시도해 보조배터리 전원 인가 후 광고 지연을 커버.
    """
    target_addr = (target_addr or "").lower()
    for attempt in range(1, tries+1):
        if STOP_EVENT and STOP_EVENT.is_set():
            return None
        devs = await BleakScanner.discover(timeout=scan_seconds)
        found = None
        for d in devs:
            name_ok = (d.name or "").strip().startswith((target_name or "").strip())
            addr_ok = (d.address or "").lower() == target_addr if target_addr else False
            if name_ok or addr_ok:
                found = d
                break
        if found:
            print(f"[SCAN] found {target_name} ({found.address}) on attempt {attempt}")
            return found
        wait = min(2.0 * attempt, 6.0)  # 점진 backoff
        print(f"[SCAN] {target_name} not found (try {attempt}/{tries}) → wait {wait:.1f}s")
        await asyncio.sleep(wait)
    return None

# ---------------------- service loop ----------------------
async def run_service(name_l, name_r, user_id,
                      mac_l=None, mac_r=None,
                      imu_master="L",
                      pair_lag_s=0.35, stale_sec=None, pain_mode=None,
                      debug=False, identity_scale=False):

    scaler_mt = load_scaler(MT_SCALER_PATH, MT_SCALER_NPZ)
    _ = load_scaler(FI_SCALER_PATH, FI_SCALER_NPZ)
    mt_model  = load_mt_model()
    _ = load_fi_model()

    # 결과 TSV 준비
    new_pred = not os.path.exists(OUT_TSV)
    fp_pred  = open(OUT_TSV, "a", newline="")
    wr_pred  = csv.writer(fp_pred, delimiter="\t")
    if new_pred:
        wr_pred.writerow(["ts","user_id","rep_id","FI_L","FI_R","AIF","AI_RMS","AI_iEMG","BI",
                          "stage_L","stage_R","BI_stage","BI_text"])

    # IMU TSV 준비
    new_imu = not os.path.exists(IMU_TSV)
    fp_imu  = open(IMU_TSV, "a", newline="")
    wr_imu  = csv.writer(fp_imu, delimiter="\t")
    if new_imu:
        wr_imu.writerow([
            "ts_unix","user_id","side","ts_ms",
            "imu_state_num","imu_state","rep_id",
            "desc_ms","rise_ms","tempo_cv","tempo_score","tempo_level",
            "pitch_deg","pitch_vel_dps"
        ])

    L = SideEngine("L"); R = SideEngine("R")
    PHASE_STR = { -1:"DESC", 0:"HOLD", 1:"RISE" }

    # --- 장치 탐색(이름 우선) ---
    devL = await resolve_device(name_l, mac_l, tries=8, scan_seconds=4.0)
    if not devL:
        raise RuntimeError(f"{name_l} ({mac_l or '-'}) 광고를 찾지 못했습니다.")
    devR = await resolve_device(name_r, mac_r, tries=8, scan_seconds=4.0)
    if not devR:
        raise RuntimeError(f"{name_r} ({mac_r or '-'}) 광고를 찾지 못했습니다.")

    async def attach(client: BleakClient, side:SideEngine):
        def cb(_, data:bytes):
            if not data: return
            tag = data[0]
            if tag==0x45 and len(data)>=10:
                _,seq,ts,fs,n = struct.unpack_from('<BBIHH', data, 0)
                n = min(n, (len(data)-10)//2)
                if n<=0: return
                samples = struct.unpack_from('<'+'h'*n, data, 10)
                side.feed_emg(samples)

            elif tag==0x49 and len(data)>=(1+4+4+4+1+2+2+2+4):
                _t, ts_ms, pitch, pitch_vel, state, rep_id, desc_ms, rise_ms, tempo_cv = \
                    struct.unpack_from('<BIffbHHHf', data, 0)
                side.feed_imu(ts_ms, pitch, pitch_vel, state, rep_id, desc_ms, rise_ms, tempo_cv)

                # imu_tempo.tsv 즉시 기록
                state_str = PHASE_STR.get(int(state), "HOLD")
                score = tempo_score_from_cv(tempo_cv)
                level = tempo_level_from_score(score)
                wr_imu.writerow([
                    r2(time.time(),3), user_id, side.name, int(ts_ms),
                    int(state), state_str, int(rep_id),
                    int(desc_ms), int(rise_ms), r2(tempo_cv,3), int(score), level,
                    r2(pitch,2), r2(pitch_vel,2)
                ])
                fp_imu.flush()

        await client.start_notify(UUID_TX, cb)

    # --- 연결 & 실행 ---
    async with BleakClient(devL) as cl, BleakClient(devR) as cr:
        print(f"[BLE] connected L={devL.address}  R={devR.address}")
        await attach(cl, L); await attach(cr, R)

        try:
            while True:
                if STOP_EVENT and STOP_EVENT.is_set():
                    break
                await asyncio.sleep(HOP/FS)

                pl = L.process(); pr = R.process()
                if not (pl and pr):  # 둘 다 RUN 상태에서 윈도 준비되어야 함
                    continue

                # 동기화 게이트(초 단위)
                ts_diff = abs(pl["ts_unix"] - pr["ts_unix"])
                if ts_diff > pair_lag_s:
                    if debug:
                        print(f"[SKIP] ts diff={ts_diff:.3f}s (limit={pair_lag_s:.3f}s) "
                              f"Lf={pl['frame_id']} Rf={pr['frame_id']}")
                    continue

                # 마스터(IMU 있는 쪽) 기준 rep/cv
                rep_id = pl["rep_id"] if imu_master=="L" else pr["rep_id"]
                tempo_cv_master = pl["tempo_cv"] if imu_master=="L" else pr["tempo_cv"]

                # 멀티태스크 입력 생성
                xm  = build_mt_features(pl, pr, tempo_cv_master)

                # ───────────────────────── 추론 ─────────────────────────
                if mt_model and mt_model.ok():
                    want = int(mt_model.Wb1.shape[0])
                    xmt  = align_to_model_dim(xm, want)[None,:]
                    xmt  = _finite(xmt)

                    # (A) 스케일러/모델 차원 진단
                    try:
                        sm = getattr(scaler_mt, "mean_", None)
                        ss = getattr(scaler_mt, "scale_", None)
                        if debug:
                            print("[CHK] model_in_dim=", want,
                                  "scaler_mean_shape=", (None if sm is None else sm.shape),
                                  "scaler_scale_shape=", (None if ss is None else ss.shape))
                    except Exception:
                        pass

                    # (B) 스케일링 (옵션 우회)
                    if identity_scale:
                        xmt_scaled = align_to_model_dim(xm, want)
                    else:
                        xmt_scaled = scaler_mt.transform(xmt)[0]
                    # ★ 스케일 결과 비정상(너무 큼/NaN/Inf/AllZero)이면 항등으로 우회
                    if (not np.isfinite(xmt_scaled).all()) or np.allclose(xmt_scaled, 0.0, atol=1e-7) \
                    or (np.nanmax(np.abs(xmt_scaled)) > 5):   # 임계치(예: 15)
                        print("[WARN] scaler mismatch → using IDENTITY features")
                        xmt_scaled = align_to_model_dim(xm, want)
                    # (C) 스케일 결과 가드: NaN/Inf/AllZero → 항등으로 전환
                    if (not np.isfinite(xmt_scaled).all()) or np.allclose(xmt_scaled, 0.0, atol=1e-7):
                        print("[WARN] scaler produced invalid/zero features → using IDENTITY scaler")
                        xmt_scaled = align_to_model_dim(xm, want)

                    if debug:
                        print("[DBG] xmt(min,max)=", float(np.min(xmt)),
                              "scaled(min,max)=", float(np.min(xmt_scaled)), float(np.max(xmt_scaled)))
                        print("[DBG] fi_raw=", mt_model(xmt_scaled.copy()))

                    FI_L, FI_R, BI = mt_model(xmt_scaled)

                else:
                    # 규칙식 Fallback (per-side)
                    def _fi_rule_from(p):
                        return rule_fi_from_norm(p["dmsesen"], p["dsampen"], p["dmdf"],
                                                 p["rms_norm"], cv_norm(tempo_cv_master))
                    FI_L, FI_R = _fi_rule_from(pl), _fi_rule_from(pr)
                    # BI는 아래 공통부에서 계산/클리핑

                # 공통 지표/출력
                ts_out = (pl["ts_unix"]+pr["ts_unix"])/2.0

                diff_rms  = abs(pl["rms_norm"] - pr["rms_norm"])
                diff_iEMG = abs(pl["diemg"]    - pr["diemg"])
                AI_RMS  = float(np.clip(diff_rms,  0.0, 1.0))
                AI_iEMG = float(np.clip(diff_iEMG, 0.0, 1.0))
                AIF = abs(FI_L - FI_R)
                BI  = float(np.clip(0.4*AI_RMS + 0.4*AI_iEMG + 0.2*AIF, 0.0, 1.0))

                dir_score = 0.4*(pl["rms_norm"] - pr["rms_norm"]) + \
                            0.4*(pl["diemg"] - pr["diemg"]) + \
                            0.2*(FI_L - FI_R)

                wr_pred.writerow([
                    f"{ts_out:.3f}", user_id, rep_id,
                    r2(FI_L), r2(FI_R), r2(AIF), r2(AI_RMS), r2(AI_iEMG), r2(BI),
                    fatigue_stage(FI_L), fatigue_stage(FI_R),
                    bi_stage(BI), bi_text(BI, dir_score)
                ])
                fp_pred.flush()

                if debug:
                    print(f"[PRED] rep={rep_id}  FI_L={r2(FI_L)}  FI_R={r2(FI_R)}  "
                          f"AIF={r2(AIF)} AI_RMS={r2(AI_RMS)} AI_iEMG={r2(AI_iEMG)} BI={r2(BI)}")
                # ──────────────────────── /추론 ────────────────────────

        finally:
            try: await cl.stop_notify(UUID_TX)
            except: pass
            try: await cr.stop_notify(UUID_TX)
            except: pass
            fp_pred.close()
            fp_imu.close()

# ---------------------- CLI ----------------------
def parse_args():
    ap = argparse.ArgumentParser()
    # 이름 우선 (기본: NANO33_L / NANO33_R)
    ap.add_argument("--name-l", default="NANO33_L", help="Left device local name")
    ap.add_argument("--name-r", default="NANO33_R", help="Right device local name")
    # 선택: MAC 보조(있으면 이름 또는 MAC 일치 시 연결)
    ap.add_argument("--mac-l", default=None, help="(optional) Left MAC")
    ap.add_argument("--mac-r", default=None, help="(optional) Right MAC")

    ap.add_argument("--imu-master", choices=["L","R"], default="L",
                    help="IMU/rep을 마스터로 삼을 측 (기본 L)")
    ap.add_argument("--pair-lag-ms", type=int, default=350,
                    help="좌/우 프레임 허용 지연(ms)")

    ap.add_argument("--stale-sec", type=float, default=None)
    ap.add_argument("--pain-mode", type=str, default=None)
    ap.add_argument("--debug", action="store_true")

    ap.add_argument("--user-id", default=None)
    ap.add_argument("--user-seq", action="store_true")
    ap.add_argument("--user-prefix", default="user")

    # ★ 추가: 스케일러 우회 옵션
    ap.add_argument("--identity-scale", action="store_true",
                    help="표준화(스케일러) 우회하고 원시 특징 그대로 사용")

    return ap.parse_args()

def next_user_id(seq_file=os.path.join(BASE_DIR,"user_seq.json"), prefix="user"):
    os.makedirs(os.path.dirname(seq_file), exist_ok=True)
    n=0
    if os.path.exists(seq_file):
        try: n=int(json.load(open(seq_file,"r")).get("last",0))
        except: n=0
    n+=1
    try: json.dump({"last":n}, open(seq_file,"w"))
    except: pass
    return f"{prefix}_{n:03d}"

def main():
    global STOP_EVENT
    STOP_EVENT = asyncio.Event()   # ← 추가
    args = parse_args()
    uid = args.user_id or (next_user_id(prefix=args.user_prefix) if args.user_seq else "unknown")
    print(f"[SERVICE] user_id={uid}")
    asyncio.run(
        run_service(args.name_l, args.name_r, uid,
                    mac_l=args.mac_l, mac_r=args.mac_r,
                    imu_master=args.imu_master,
                    pair_lag_s=args.pair_lag_ms/1000.0,  # 초 단위로 변환
                    stale_sec=args.stale_sec,
                    pain_mode=args.pain_mode,
                    debug=args.debug,
                    identity_scale=args.identity_scale)
    )

if __name__ == "__main__":
    main()
