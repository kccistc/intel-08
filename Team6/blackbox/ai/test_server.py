# -*- coding: utf-8 -*-
"""
vision_server.py (테스트 더미)
- C에서 "analyze {json}\n" 요청이 오면 0.5초 뒤 임의의 객체 목록을 JSON으로 응답.
- 표준 출력(stdout)으로만 결과를 보내고, 표준 에러(stderr)로는 로그를 남김.
- 의존성: 표준 라이브러리만 사용.
"""
import sys
import json
import time
import random
import signal

def log(msg: str):
    print(f"[Py LOG] {msg}", file=sys.stderr, flush=True)

def make_dummy_objects():
    """
    C 쪽 DetectedObject 스키마에 맞춘 더미 객체들 생성:
      - label: 0~4
      - x, y: 전방/좌우 위치 (m 가정) -> 대략 1~30m, -5~5m
      - ax, ay: 상대 가속/속도 같은 값 느낌으로 -2~2 범위 임의
    """
    n = random.randint(1, 5)  # 0~5개
    objs = []
    for _ in range(n):
        obj = {
            "label": random.randint(0, 4),
            "x": round(random.uniform(1.0, 30.0), 2),
            "y": round(random.uniform(-5.0, 5.0), 2),
            "ax": round(random.uniform(-2.0, 2.0), 2),
            "ay": round(random.uniform(-2.0, 2.0), 2),
        }
        objs.append(obj)
    return objs

def parse_command(line: str):
    """
    'analyze {json}' 형태에서 payload만 파싱 (없어도 동작은 하게 관대하게 처리)
    """
    line = line.strip()
    if not line:
        return None, None
    if not line.startswith("analyze"):
        return None, None

    payload = None
    # 'analyze ' 뒤에 JSON이 붙어 있을 수도 있음
    if len(line) > len("analyze"):
        rest = line[len("analyze"):].strip()
        if rest:
            try:
                payload = json.loads(rest)
            except Exception:
                payload = None
    return "analyze", payload

def main():
    random.seed()  # 필요하면 고정 seed로 재현성 확보 가능: random.seed(1234)
    log("Dummy vision server started. Waiting for commands on stdin...")

    # Ctrl+C 핸들러(깨끗한 종료)
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))

    while True:
        line = sys.stdin.readline()
        if not line:
            log("stdin closed. exiting.")
            break

        cmd, payload = parse_command(line)
        if cmd != "analyze":
            log(f"ignored line: {line.strip()}")
            continue

        # (옵션) 넘겨받은 GPS/steer를 참고해 무언가 하려면 payload를 활용
        # payload 예: {"gps":[x,y], "steer": deg}
        log(f"received analyze request. payload={payload}")

        # 요구사항: 0.5초 뒤에 임의 데이터 응답
        time.sleep(0.5)

        result = {
            "status": "ok",
            "objects": make_dummy_objects(),
        }
        print(json.dumps(result, ensure_ascii=False))
        sys.stdout.flush()
        log("dummy result sent.")

if __name__ == "__main__":
    main()
