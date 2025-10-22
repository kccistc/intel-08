import os, time, threading, glob, datetime
from collections import deque
from pathlib import Path
import cv2
import numpy as np
import queue # 'import queue'를 위로 옮겼습니다.

# 전역 상수 대신 __init__에서 self.num_cams = 6으로 설정합니다.
ENC_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), 80] # 메모리 절약

def _ts_str():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

class TimeWindowEventRecorder6:
    """
    - 6캠 동시 배치 또는 1캠 단일 프레임을 받아 저장
    - trigger(tag) 호출 시: t0=현재시간 기준 [t0- pre_secs, t0 + post_secs] 윈도우 저장
    - FPS 의존 없음 (전적으로 ts 기반)
    """

    def __init__(self, out_dir="event6", size=(800, 450), pre_secs=5.0, post_secs=5.0,
                 retention_secs=60.0, # 버퍼에 보관할 최대 시간 (메모리 제한용)
                 save_as="jpg", # "jpg" | "mp4"
                 fourcc_str="mp4v",
                 target_fps=5.0, # ← 추가: 저장/표시 FPS 고정
                 exact_count=True):
        
        self.num_cams = 6 # 🌟 변경: 전역 상수 대신 멤버 변수 사용
        
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.size = tuple(size)
        self.pre_secs = float(pre_secs)
        self.post_secs = float(post_secs)
        self.retention_secs = float(retention_secs)
        self.save_as = save_as
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        self.target_fps = float(target_fps)
        self.exact_count = bool(exact_count)
        
        # 시간 기반 링버퍼: [(ts, [Optional[jpg_bytes_cam0]..cam5])]
        self.buffer = deque()
        self._lock = threading.Lock()
        
        # 진행 중 이벤트(단일 세션)
        self.active = False
        self.t0 = None
        self.event_id = None
        self._post_done = False
        
        # exact_count 모드에서 사용: 필요/수집 개수
        self._pre_needed = 0
        self._post_needed = 0
        self._post_got = 0
        
        # 비동기 저장 워커 (블로킹 방지)
        self._jobs: "queue.Queue[tuple[list, str]]" = queue.Queue(maxsize=8)
        self._stop = False
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    # ------------- 외부에서 호출하는 API -------------

    def push_batch(self, frames, ts=None):
        """
        frames: 길이 6의 BGR np.ndarray 리스트
        ts: float epoch seconds (None이면 time.time())
        """
        if ts is None:
            ts = time.time()
            
        # 🌟 변경: NUM_CAMS -> self.num_cams
        assert len(frames) == self.num_cams, f"frames must be length-{self.num_cams}"
        
        # 리사이즈 & JPEG 인코딩
        jpgs = []
        for fr in frames:
            if fr is None: # 🌟 추가: 혹시 None이 들어오면
                jpgs.append(None)
                continue
                
            if fr.shape[1] != self.size[0] or fr.shape[0] != self.size[1]:
                fr = cv2.resize(fr, self.size)
            ok, jb = cv2.imencode(".jpg", fr, ENC_PARAMS)
            
            # 🌟 변경: 인코딩 실패 시 None 추가 (전체 배치를 버리지 않음)
            if ok:
                jpgs.append(jb.tobytes())
            else:
                jpgs.append(None)

        if len(jpgs) != self.num_cams:
            return # 이 경우는 assert로 이미 걸러짐
            
        # 🌟 추가: 모든 프레임이 None이면 버퍼에 추가하지 않음
        if all(jb is None for jb in jpgs):
            return

        self._add_to_buffer_locked(ts, jpgs)

    # 🌟🌟🌟 추가: 요청하신 단일 푸시 함수 🌟🌟🌟
    def push_single(self, frame, cam_id=0, ts=None):
        """
        단일 카메라 프레임을 버퍼에 추가합니다.
        
        frame: BGR np.ndarray
        cam_id: 이 프레임이 속한 카메라 번호 (0~5)
        ts: float epoch seconds (None이면 time.time())
        """
        if ts is None:
            ts = time.time()

        if not (0 <= cam_id < self.num_cams):
            print(f"[WARN] Invalid cam_id: {cam_id}. Must be 0-{self.num_cams-1}")
            return
            
        if frame is None:
            return

        # 리사이즈 & JPEG 인코딩
        if frame.shape[1] != self.size[0] or frame.shape[0] != self.size[1]:
            frame = cv2.resize(frame, self.size)
        
        ok, jb = cv2.imencode(".jpg", frame, ENC_PARAMS)
        if not ok:
            print("[WARN] Failed to encode single frame")
            return

        # 🌟 핵심: 6개짜리 리스트를 만들되, 해당 cam_id에만 데이터를 넣고 나머지는 None
        jpgs = [None] * self.num_cams
        jpgs[cam_id] = jb.tobytes()

        self._add_to_buffer_locked(ts, jpgs)

    # 🌟 추가: push_batch와 push_single의 공통 로직 추출
    def _add_to_buffer_locked(self, ts, jpgs_list):
        """버퍼에 데이터를 추가하고 트리거 로직을 처리하는 내부 함수"""
        with self._lock:
            self.buffer.append((ts, jpgs_list))
            self._prune_old_locked(ts)
            
            if self.active and self.exact_count:
                # t0 이후 프레임 개수 카운팅
                if ts > self.t0:
                    self._post_got += 1
                # 이후 프레임이 목표 개수에 도달하면 finalize
                if self._post_got >= self._post_needed:
                    self._finalize_and_enqueue_locked()
            elif self.active and not self.exact_count:
                # (구) 시간 기반 모드가 필요하면 여기서 처리할 수 있음
                if ts >= self.t0 + self.post_secs:
                    self._finalize_and_enqueue_locked()

    def trigger(self, tag: str = ""):
        """이벤트 시작(현재 시간을 기준으로 pre/post 윈도우 결정)"""
        with self._lock:
            if self.active:
                return # 단일 세션 정책: 이미 진행 중이면 무시(동시 다중 필요 시 확장)
            
            self.active = True
            self.t0 = time.time()
            self.event_id = f"{_ts_str()}{('_'+tag) if tag else ''}"
            
            # exact_count 모드: 프레임 개수로 선/후롤 목표 설정
            if self.exact_count:
                self._pre_needed = int(round(self.pre_secs * self.target_fps))
                self._post_needed = int(round(self.post_secs * self.target_fps))
                self._post_got = 0 # t0 이후로 들어오는 배치 개수

    def close(self, wait: float = 5.0):
        self._stop = True
        t0 = time.time()
        while not self._jobs.empty() and (time.time() - t0 < wait):
            time.sleep(0.05)
        self._worker.join(timeout=max(0.0, wait - (time.time() - t0)))

    # ------------- 내부 유틸 -------------
    # (변경 없음)
    def _prune_old_locked(self, now_ts):
        cutoff = now_ts - self.retention_secs
        while self.buffer and self.buffer[0][0] < cutoff:
            self.buffer.popleft()

    # (변경 없음)
    def _take_last_with_pad(self, seq, n, pad_with=None):
        if n <= 0: return []
        if len(seq) >= n: return seq[-n:]
        if not seq:
            if pad_with is None: return []
            return [pad_with] * n
        need = n - len(seq)
        pad = [ (seq[0] if pad_with is None else pad_with) ] * need
        return pad + seq

    # (변경 없음)
    def _take_first_with_pad(self, seq, n, pad_with=None):
        if n <= 0: return []
        if len(seq) >= n: return seq[:n]
        if not seq:
            if pad_with is None: return []
            return [pad_with] * n
        need = n - len(seq)
        pad = [ (seq[-1] if pad_with is None else pad_with) ] * need
        return seq + pad

    # (변경 없음)
    def _finalize_and_enqueue_locked(self):
        t0 = self.t0
        if self.exact_count:
            buf = list(self.buffer)
            pre = [b for b in buf if b[0] <= t0]
            post = [b for b in buf if b[0] > t0]
            pad_for_pre = post[0] if post else (pre[-1] if pre else None)
            pad_for_post = pre[-1] if pre else (post[0] if post else None)
            pre_sel = self._take_last_with_pad(pre, self._pre_needed, pad_with=pad_for_pre)
            post_sel = self._take_first_with_pad(post, self._post_needed, pad_with=pad_for_post)
            window = pre_sel + post_sel
        else:
            pre_from = t0 - self.pre_secs
            post_to = t0 + self.post_secs
            window = [b for b in self.buffer if pre_from <= b[0] <= post_to]
            window.sort(key=lambda x: x[0])

        if not window:
            self._reset_event_locked()
            return
            
        event_dir = str(self.out_dir / self.event_id)
        try:
            self._jobs.put_nowait((window, event_dir))
        except Exception:
            pass
        self._reset_event_locked()

    # (변경 없음)
    def _reset_event_locked(self):
        self.active = False
        self.t0 = None
        self.event_id = None
        self._pre_needed = 0
        self._post_needed = 0
        self._post_got = 0

    def _worker_loop(self):
        # import queue # (위로 이동)
        while not self._stop or not self._jobs.empty():
            try:
                window, event_dir = self._jobs.get(timeout=0.1)
            except queue.Empty:
                continue
                
            try:
                self._save_window(window, event_dir)
            except Exception as e:
                print("[SAVE][ERR]", e)
            finally:
                self._jobs.task_done()

    def _save_window(self, window, event_dir):
        os.makedirs(event_dir, exist_ok=True)
        
        # 🌟 변경: NUM_CAMS -> self.num_cams
        cam_streams = {i: [] for i in range(self.num_cams)} # list[(ts, jpg_bytes)]
        
        # 🌟 변경: jpg6 -> jpgs
        for ts, jpgs in window:
            # 🌟 변경: NUM_CAMS -> self.num_cams
            for i in range(self.num_cams):
                jb = jpgs[i]
                # 🌟🌟🌟 핵심: None이 아닌 데이터만 스트림에 추가 🌟🌟🌟
                if jb is not None:
                    cam_streams[i].append((ts, jb))

        if self.save_as == "jpg":
            # 🌟 변경: NUM_CAMS -> self.num_cams
            for cam in range(self.num_cams):
                # 🌟 추가: 이 부분은 이미 원본 코드에 있었지만,
                # None 처리 로직 덕분에 이제 정상 동작합니다.
                if not cam_streams[cam]:
                    continue # 저장할 프레임이 없는 캠은 건너뛰기

                cdir = Path(event_dir) / f"cam{cam}"
                cdir.mkdir(parents=True, exist_ok=True)
                for idx, (ts, jb) in enumerate(cam_streams[cam]):
                    with open(cdir / f"{idx:04d}_{int(ts*1000)}.jpg", "wb") as f:
                        f.write(jb)
        else:
            # "mp4"
            duration_s = (self.pre_secs + self.post_secs) # 10.0
            
            # 🌟 변경: NUM_CAMS -> self.num_cams
            for cam in range(self.num_cams):
                items = cam_streams[cam]
                if not items:
                    continue # 저장할 프레임이 없는 캠은 건너뛰기

                # (이하 로직은 변경 없음)
                first = items[0][1]
                img0 = cv2.imdecode(np.frombuffer(first, dtype=np.uint8), cv2.IMREAD_COLOR)
                h, w = img0.shape[:2]
                
                N = len(items)
                fps_out = max(1.0, min(30.0, float(N) / max(0.001, duration_s)))
                
                vw_path = str(Path(event_dir) / f"cam{cam}.mp4")
                vw = cv2.VideoWriter(vw_path, self.fourcc, fps_out, (w, h))
                if not vw.isOpened():
                    print(f"[ERR] VideoWriter open failed: {vw_path}")
                    continue # raise 대신 다음 캠으로
                    
                for _, jb in items:
                    fr = cv2.imdecode(np.frombuffer(jb, dtype=np.uint8), cv2.IMREAD_COLOR)
                    vw.write(fr)
                vw.release()

            print(f"[SAVE] {event_dir} frames_total={len(window)} "
                  f"→ fixed_duration={duration_s:.2f}s (variable FPS)")