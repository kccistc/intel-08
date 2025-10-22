# 🏋️‍♂️ **AI Smart Gym Project**



## 📘 **프로젝트 개요 (Overview)**

**AI Smart Gym**은 포즈 랜드마크 기반 운동 분류 + 실시간 분석 앱이며,  
EMG·IMU 센서 융합 파워리프팅 스쿼트 분석까지 지원하는 파이썬 중심 프로젝트입니다.  

애플리케이션은 **Raspberry Pi**에서 구동되며,  
카메라·센서 스트림을 받아 운동을 인식하고 자세·가동범위(ROM)·템포·피로도·하체불균형 등 핵심 지표를 실시간으로 제공하고 피드백 해줍니다.

---

## 🎯 **프로젝트 목표**

- 운동 동작의 **정확한 분류 및 실시간 분석**  
- IMU, EMG 등 센서를 통한 **정량적 운동 데이터 수집**  
- AI 기반 **운동 수행 평가 알고리즘 및 피드백 제공**  
- **Raspberry Pi / Hailo-8 경량화 및 실시간 처리**

---

## 🧩 **구현 범위 (Minimum Viable Product)**

| 항목 | 내용 |
| --- | --- |
| **목표** | RPi5 + Hailo-8에서 실시간 스쿼트 카운트·점수·요약 제공 |
| **입력** | 측면 카메라 (1인 대상) |
| **처리** | YOLOv8-Pose → 각도(무릎/힙) → 카운트 FSM → 점수 |
| **출력** | PySide6 오버레이 UI, 세션 리포트(SQLite) |
| **성능 목표** | FPS ≥ 15 / 지연 < 250ms / 카운트 정확도 ≥ 90% |
| **비범위** | 다중 인원, EMG/클라우드/3D 분석 |
| **위험요소** | 런타임 중 프레임 드롭 |
| **데모 시나리오** | 3회 수행 → 실시간 표시 → 요약 확인 |

---

## ✅ **구현 (Done)**

- [x] 카메라 입력 파이프라인 (OpenCV, 고정 해상도)  
- [x] YOLOv8-Pose(ONNX) 추론 및 COCO17 키포인트 추출  
- [x] 각도 계산 모듈 (무릎/힙) + 안정화(클램프·스무딩)  
- [x] 7개 운동 카운트 로직 설계  
- [x] 점수 매핑 (Good/OK/Retry) + 오버레이 UI (PySide6)  
- [x] HailoCamAdapter API (`frame()/people()/meta`)  
- [x] 세션 요약 저장 (SQLite 최소 컬럼)  
- [x] 데모 스크립트 (3회 수행 → 요약 확인)  
- [x] EMG/IMU 센서 데이터 수집 (Arduino → Pi5 Bluetooth)  
- [x] 근육 피로도/불균형 모델 아키텍처 구현 및 프로토타입 생성  
- [ ] 품질관리 기준 데이터셋 수집 — 센서 공급 지연  
- [ ] Yocto 빌드 — 디렉토리 경로 불일치 오류 발생  

---

## ⚙️ **주요 기능**

### 🔹 운동 분류 모델 (AI)
- **TCN 기반 동작 분류 모델**
- 포즈 추정 + 센서 데이터 결합
- ONNX / HEF 기반 경량 추론 및 시각화

### 🔹 운동 분석 알고리즘
- 속도·가속도·파워 등 **운동 성능 지표 계산**
- 센서 기반 **운동 품질 평가 로직 설계**

### 🔹 센서 하드웨어 수집
- IMU, EMG 등 **실시간 수집**
- BLE/UART 통신, ADC,노이즈 필터링·캘리브레이션

### 🔹 통합 어플리케이션
- AI + 알고리즘 + 센서 모듈 통합
- **PySide6 UI 시각화 + BLE/Wi-Fi 데이터 연동**

---

## 🖥️ **시스템 구성도**

<img width="839" height="430" alt="image" src="https://github.com/user-attachments/assets/97e1d978-0dcd-4460-b81d-8fe5502806a3" />
<img width="861" height="485" alt="image" src="https://github.com/user-attachments/assets/eeaca896-b994-43b9-892b-fadff9362e0d" />



---

## 👥 **Team: 자세어때**

| 이름 | 역할 | 주요 담당 |
| --- | --- | --- |
| **서민솔** | 팀장 | 프로젝트 총괄, 운동 분류 모델 설계 |
| **이동현** | 부팀장 | 통합 어플리케이션 개발 |
| **유종민** | 센서 | 센서 신호처리 AI 개발, 3D 모델링 |
| **윤찬민** | AI 개발 | 운동 분류 모델 구현 |
| **임정민** | 운동 분석 | 운동 분석 알고리즘 개발,Yocto 개발 |

---

## 🧠 **기술 스택**

| 분야 | 기술 |
| --- | --- |
| **AI / ML** | PyTorch, ONNX, TCN |
| **임베디드** | Arduino, Raspberry Pi, Hailo-8 |
| **센서** | IMU, EMG |
| **프론트엔드 / 앱** | PySide6, BLE 통신, Python |
| **기타** | YAML Config, CSV/JSON Logging, Autodesk Fusion 360 |

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="48" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytorch/pytorch-original.svg" width="48" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/raspberrypi/raspberrypi-original.svg" width="48" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/arduino/arduino-original.svg" width="48" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/opencv/opencv-original.svg" width="48" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linux/linux-original.svg" width="48" />
  <img src="https://cdn.jsdelivr.net/npm/simple-icons@v11/icons/qt.svg" width="48" title="PySide6 (Qt for Python)"/>
  <img src="https://cdn.jsdelivr.net/npm/simple-icons@v11/icons/onnx.svg" width="48" title="ONNX Runtime"/>
  <img src="https://cdn.jsdelivr.net/npm/simple-icons@v11/icons/autodesk.svg" width="48" title="Autodesk Fusion 360"/>
  <a href="https://hailo.ai/" title="Hailo">
    <img src="https://img.shields.io/badge/Hailo-000?style=for-the-badge" height="24"/>
  </a>
</p>
---

## 🚀 **기대 효과**

- 운동 수행 정확도 향상 및 **부상 예방**  
- 개인 맞춤 피드백을 통한 **훈련 효율 극대화**  
- AI + 센서 융합 **스마트 피트니스 솔루션 실현**

---

## 🎬 **시연 예시**

<p align="center">
  <img src="assets/demo.gif" alt="Demo" width="70%">
</p>

---

## 🧩 **Clone Code**
git clone https://github.com/Biomedical-Signal-Processing-Lab/smart_gym_project.git


## ⚙️ **Steps to Build**

```
# 0) 기본 설정
sudo apt update
sudo apt install -y git curl wget build-essential pkg-config
python -m venv .sgym_venv
source .sgym_venv/bin/activate
cd smart_gym_project/app
pip install -r requirements.txt

# 1) Hailo (공식 APT 레포 추가 후 설치)
# ⚠️ 반드시 벤더 문서 절차에 따라 레포를 먼저 등록해야 합니다.
sudo apt install -y hailo-all

# 2) GStreamer 런타임 + 플러그인 묶음
sudo apt install -y \
  gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav \
  gstreamer1.0-gl gstreamer1.0-alsa

# 3) GI(PyGObject) 바인딩 (Python에서 GStreamer를 사용하는 경우)
sudo apt install -y \
  python3-gi python3-gi-cairo gobject-introspection \
  gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 libgirepository1.0-dev

# 4) 카메라 유틸리티 설치
sudo apt install -y v4l-utils libcamera-apps

```
## ▶️ **Step to Run**
```
# 1) 가상환경 활성화
source .sgym_venv/bin/activate

# 2) 프로젝트 실행
python main.py


---

> 💡 **Tip:**  
> 첫 실행 시 `.venv` 환경을 다시 활성화해야 합니다:  
> ```bash
> source .sgym_venv/bin/activate
> 
> 실행 후 UI 창이 뜨면, 센서 연결 상태와 카메라 입력이 정상 동작하는지 로그를 확인하세요.

---

# 3) 카메라 유틸리티 설치
sudo apt install -y v4l-utils libcamera-apps
```

## 🧩 **운동 분류 시스템**

운동 분류 시스템은 **포즈 랜드마크 추출 → 시퀀스 분류 → (상세 분석) → (후속 처리)** 순서로 동작합니다.


### **1. 영상 입력·전처리 (Raspberry Pi 5 + Hailo-8)**

- 프레임 리사이즈  
- 색공간 변환  

### **2. 포즈 키포인트 추출**

- **YOLOv8s-Pose @ Hailo-8**  
- 매 프레임 키포인트 추출  
- 키포인트 정규화: 스케일·중심  

### **3. 시퀀스 버퍼링**

- 윈도우: **60 프레임**  
- 실시간 **stride = 1**  
- 슬라이딩 업데이트  

### **4. 운동 분류 (ONNX TCN)**

- 입력: 정규화 키포인트 시퀀스 (60 프레임)  
- 분류 클래스: **idle / shoulder_press / 덤벨로우 / 점핑잭 / 스쿼트 / 푸쉬업 / 레그레이즈 / 버피 / 사이드 레터럴 레이즈**  
- (옵션) 히스테리시스·스무딩  

#### **후속 처리 (운동 채점)**

- 각도 기반 공통 채점: 각 운동 자세 채점에 필요한 관절만 사용  
- 가중치·진행률: 관절별 가중치, 목표 각도 범위 대비 progress(0~1) 계산  
- 점수 산출: 한 동작의 인식 가동 범위 내에서 최고 혹은 최저 각으로 산출  

---
## 🧠 근전도·IMU 기반 운동 상세 분석 시스템

운동 분석 시스템은 **센서 데이터 수집 → 전처리 → 전송 → 신호 분석 → AI 추론** 단계로 동작합니다.  
본 시스템은 **좌·우 허벅지의 근전도(EMG)** 와 **IMU 센서 데이터**를 융합하여  
운동 중 **피로도(Fatigue Index, FI)** 와 **불균형도(Balance Index, BI)**,  
그리고 **템포 일관성(Tempo Consistency)** 을 분석합니다.


### 1. 데이터 수집
- **Arduino Nano 33 IoT (2대)** 사용  
- 좌/우 허벅지 근전도(EMG) 센서 각 1개 연결  
- IMU(가속도계/자이로)로 하강·상승 동작 구분 및 템포 계산  
- 샘플링 속도: EMG 500Hz, IMU 10Hz


### 2. 전처리
- Arduino 단에서 **EMG 필터링 및 정규화(DC offset 제거)**  
- IMU를 이용한 **하강/상승 구간 분리 및 템포 추출**  
- 3초간 MVC(Maximum Voluntary Contraction) 측정을 통한 **근수축 기준 정규화**


### 3. 데이터 전송
- BLE(Bluetooth Low Energy)를 이용해 라즈베리파이로 전송  
- 좌측(NANO33_L), 우측(NANO33_R) 장치로 구분  
- 전송 데이터:  
  - `EMG_preprocessed`, `IMU_pitch`, `tempo`, `rep_id`


### 4. 신호 분석
- Raspberry Pi에서 실시간 분석 수행  
- FFT 기반 주파수 도메인 특징 및 시간 도메인 특징 추출  

| 특징값 | 의미 | 피로 시 변화 |
|:---|:---|:---|
| **RMS_norm** | 근육 수축 세기(정규화 RMS) | ⬇️ 감소 |
| **MDF** | 스펙트럼 중심 주파수 | ⬇️ 저주파 쪽 이동 |
| **SampEn** | 신호의 불규칙성(복잡도) | ⬇️ 더 규칙적 |
| **MSESEn** | 여러 시간 스케일에서의 복잡도 | ⬇️ 장·단기 패턴 단순화 |
| **iEMG_norm** | EMG 적분값(활동량) | ⬇️ 활성 근섬유 감소 |
| **tempo_cv** | 스쿼트 템포의 변동성(속도 일관성 지표) | ⬆️ 증가 시 불균일한 리듬 |


### 5. AI 추론
#### 🧩 멀티태스크 AI 모델 (Multi-Task MLP)
- 입력: `[RMS_norm, MDF, SampEn, MSESEn, iEMG_norm, tempo_cv]`
- 출력:
  - **FI_pred**: 피로도 (Fatigue Index, 0~1)
  - **BI_pred**: 불균형도 (Balance Index, 0~1)
- 구조:
  - Dense(32) → ReLU → Dense(16) → 공유층  
  - Head1(FI): Dense(8) → Sigmoid  
  - Head2(BI): Dense(8) → Sigmoid  
- 손실 함수:  
  `Loss = α·MSE(FI_pred, FI_label) + β·MSE(BI_pred, BI_label)`

#### ⚖️ 보조 계산 지표
| 항목 | 계산식 | 의미 |
|:---|:---|:---|
| **AIF** | \|FI_L - FI_R\| | 피로 누적 불균형 |
| **AI_RMS** | \|RMS_L - RMS_R\| | 좌·우 근수축 세기 차이 |
| **AI_iEMG** | \|iEMG_L - iEMG_R\| | 좌·우 근육 활성 차이 |
| **BI (최종)** | 0.4×AI_RMS + 0.4×AI_iEMG + 0.2×AIF | 종합 불균형 점수 |


### 6. 출력 및 시각화
- **FI, BI, tempo_cv** 값을 실시간으로 시각화  
- PyQt 대시보드에서 게이지/그래프 형태로 표시  
- 결과는 `.tsv` 형식(`window_features.tsv`, `reps_pred_dual.tsv`)으로 자동 저장

### 7. 웨어러블 케이스 모델링
<img width="817" height="688" alt="image" src="https://github.com/user-attachments/assets/db870a0e-abe2-4cee-9077-f69e6831b411" />

### 8. 웨어러블 센서 기기 착용 방식
<img width="694" height="938" alt="image" src="https://github.com/user-attachments/assets/7d32eb56-e8d7-4581-b838-29756dc52e43" />



### 🧠 요약
- **FI (Fatigue Index)** → 근육 피로 누적 정도  
- **BI (Balance Index)** → 좌우 근육 사용의 불균형 정도  
- **tempo_cv** → 스쿼트 속도의 일관성 (리듬 안정성)
- 세 지표를 통해 운동자의 **피로도, 균형, 리듬**을 동시에 평가합니다.



### **기대 효과**

- 운동 수행 정확도 향상 및 부상 예방  
- 개인 맞춤형 피드백을 통한 훈련 효율 극대화  
- AI + 센서 융합을 통한 스마트 피트니스 솔루션 실현  
- 실시간 채점 및 시각화로 재미 향상  

---

## ⚙️ **시행착오 및 해결방안**

| **No.** | **시행착오** | **해결 방안** | **결과 / 교훈** |
| :---: | --- | --- | --- |
| 1 | 프로토타입(**MediaPipe**) ↔ 배포(**Hailo-8, COCO-17**) **포즈 스키마 불일치**로 입력 붕괴 | **COCO-17 기준 전 파이프라인 정렬** (키포인트 매핑 / 좌표·정규화 통일) + 데이터·레이블 **재생성·재학습** | 배포 타깃 스키마 **데이터 계약** 고정 · 스키마 변경 시 **어댑터·회귀 테스트·버전 태깅** 필수 |
| 2 | 오프라인 지표 우수 ↔ 실사용에서 **숄더프레스·덤벨로우 혼동** (유사 패턴, 시간 특성 미활용 / 지름길) | 정규화 키포인트에 **속도·가속도 (1차/2차 차분)** 채널 추가 → **TCN 입력 다채널화** (윈도우 60, stride 1) | 지름길 억제 · 유사 클래스 **분리도↑** · 경계 흔들림 **완화** (히스테리시스 / 스무딩) |
| 3 | 기본 카메라 **FOV 협소**로 공간 제약·스케일 변동 | **광각 카메라 전환**, **왜곡 보정 없이** 광각 전용 데이터 **재수집·재학습** | 배포 **광학 스펙을 데이터 계약**으로 고정 · 보정 미적용 시 **훈련=추론 조건 일치** 유지 |

---

## 💬 **개발 후기**

AI Smart Gym 프로젝트는 라즈베리파이5와 Hailo-8을 이용해 실시간으로 스쿼트 자세를 인식하고 피드백을 주는 시스템이다.
카메라로 포즈를 추적하고, EMG·IMU 센서로 근육 피로도와 좌우 균형을 분석했다.
초기엔 포즈 데이터 불일치나 센서 연결 문제 등 시행착오가 많았지만, 포맷 통일과 속도·가속도 정보를 추가하면서 정확도를 높였다.
결과적으로 실시간 분석 속도와 안정성이 확보됐고, 데이터 기반으로 운동 품질을 평가하는 구조를 완성했다.
앞으로는 EMG 자동 보정과 리포트 기능을 개선해 완성도를 높일 계획이다.

---

## 📎 **Appendix**

[피로도 분석 논문1.pdf](https://github.com/user-attachments/files/23041197/default.pdf)
[피로도 분석 논문2.pdf](https://github.com/user-attachments/files/23042535/default.pdf)
[피로도 분석 논문3.pdf](https://github.com/user-attachments/files/23042548/s41598-019-41860-4.pdf)


