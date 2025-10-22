# AI 위험판단 블랙박스 (AI Risk Detection Blackbox)

AI 기반으로 차량의 위험 상황을 실시간 분석하고, 사고 전후의 맥락을 함께 저장하는 지능형 블랙박스 시스템입니다.
단순한 영상 기록을 넘어, 차량 센서 데이터(CAN)와 AI 모델 분석을 결합해 위험도 기반의 이벤트 감지를 수행합니다.

<!-- GIF나 이미지 데모 추가 가능: ![데모 GIF](img/demo.gif) -->

# 목차
- [프로젝트 개요](#프로젝트-개요)
- [시스템 구성](#시스템-구성)
- [핵심 기술](#핵심-기술)
  - [Carla](#Carla)
  - [CAN 통신](#CAN-통신)
  - [PETR](#PETR)
  - [Yocto](#Yocto)
- [개발 환경](#개발-환경)
- [하드웨어 구성](#하드웨어-구성)
- [설치](#설치)
  - [Ubuntu-RPi 크로스 컴파일 환경 구성](#Ubuntu-RPi-크로스-컴파일-환경-구성)
  - [컴파일 방법](#컴파일-방법법)
  - [git clone](#git-clone)
  - [의존성 설치](#의존성-설치)
- [사용법](#사용법)
- [출력 구조](#출력-구조)
- [결과 및 시연](#결과-및-시연)
- [팀원 소개](#팀원-소개)

### 프로젝트 개요

기존 블랙박스는 사고가 난 이후의 영상만 기록하기 때문에,
사고 이전의 위험 징후나 차량 상태를 분석하기 어렵다는 한계가 있습니다.

이 프로젝트는 이러한 문제를 해결하기 위해
AI 기반 위험도 판단 및 사고 전후 맥락 분석이 가능한 블랙박스를 구현합니다.
  

이 프로젝트는 이러한 요구사항을 모두 충족합니다.

### 시스템 구성

|구성 요소	               |설명                                  |
|--------------------------|--------------------------------------|
|Raspberry Pi 5            |메인 컨트롤러, 영상 수집 및 전처리      |
|Hailo-8	                 |AI 추론 가속기 (PETR 모델 구동)        |
|CAN 모듈                  |차량 상태 데이터 수집 (속도, 브레이크 등)|
|CARLA                     | 시뮬레이터	가상 도로 환경에서의 테스트 |
|Yocto / AGL              | 임베디드 환경 기반 OS 및 런타임 구성    |

### 핵심 기술

------

#### Carla

<img width="650" height="324" alt="image" src="https://github.com/user-attachments/assets/7e1f7057-d635-463a-a31b-3aec24aa26e0" />


Carla의 핵심 기능


<img width="664" height="320" alt="image" src="https://github.com/user-attachments/assets/4fad1c3b-9a46-4b29-8af3-ada2b16fcf3b" />

<img width="648" height="319" alt="image" src="https://github.com/user-attachments/assets/b79ae901-1e56-40de-a2eb-cae7689f3b1b" />

--------

#### CAN 통신

CAN(Controller Area Network) 통신은 자동차, 산업 자동화 등에서 널리 사용되는 네트워크 프로토콜입니다.

표준 PID 참고 URL : https://en.wikipedia.org/wiki/OBD-II_PIDs


<img width="638" height="292" alt="image" src="https://github.com/user-attachments/assets/828597a5-f9c9-466c-9343-7cc957cd0a73" />


본 프로젝트에서의 커스텀 PID를 이용한 CAN 통신

<img width="635" height="306" alt="image" src="https://github.com/user-attachments/assets/43ceba2c-8660-4f31-84bb-6dd10e085e3a" />

----------

#### PETR

<img width="659" height="317" alt="image" src="https://github.com/user-attachments/assets/ebfe4f45-999d-407c-85ef-c158c8d48516" />

<img width="930" height="436" alt="image" src="https://github.com/user-attachments/assets/884e99f4-795b-4b14-a421-a4d76dff4133" />

- Multi-view 이미지를 Backbone을 통해 2D Feature 추출
- 3D Meshigrid 형태의 Camera Frustum Sace 생성
- Frustum을 이용하여 3D World Space Coordinate 로 변환
- 3D Position Encoder에서 3D Position Aware Feature 추출
- Object Query 와 Cross-Attention
- 최종결과(3D box + class) 예측

--------

#### Yocto 

Yocto Project는 임베디드 소프트웨어 개발을 위한 도구 체인을 제공하는 프로젝트로, 개발자들이 하드웨어에 최적화된 Linux 배포판을 쉽게 만들 수 있도록 설계되었습니다.

<img width="984" height="482" alt="image" src="https://github.com/user-attachments/assets/b9d1414e-768a-4aa1-bc6d-9c3833af3463" />

| 섹션 | 하위 요소 | 설명 |
|-------|----------|------|
| **왼쪽: 입력 및 구성 요소** | User Configuration (사용자 구성) | - Metadata (.bb + patches): 레시피 파일과 패치.<br>- Machine BSP Configuration: 타겟 하드웨어 설정.<br>- Policy Configuration: 정책 설정 (컴파일 옵션 등). |
| | Source Materials (소스 자료) | - Source Fetching: 소스 가져오기 (Upstream, Local, SCMs [optional]).<br>- Patch Application: 패치 적용.<br>- Configure / Compile / Autoconf as needed: 구성, 컴파일, Autoconf 실행. |
| **중앙: 빌드 시스템** | Output Analysis for Package Splitting plus relationships | 패키지 분할과 관계 분석 (컴파일 결과 분할 및 의존성 분석). |
| | .deb generation, .rpm generation, .ipk generation | 패키지 생성 (Debian, RPM, IPK 형식). |
| | QA Tests | 품질 보증 테스트 (오류 및 호환성 검사). |
| **오른쪽: 출력** | Package Feeds (패키지 피드) | - Image Generation: 시스템 이미지 생성.<br>- SDK Generation: SDK 생성. |
| | 최종 출력 | - Application Images: 애플리케이션 이미지.<br>- Application Development SDK: 개발 SDK. |


-동작 원리
  - 소스 자료 수집: 업스트림 소스, 로컬 프로젝트, SCM(Git 등)에서 소스를 가져옵니다.
  - 사용자 구성: local.conf, bblayers.conf 등으로 머신, 배포판, 레이어를 설정
  - 빌드 실행: BitBake가 레시피를 처리하여 패치 적용, 컴파일, 패키지 생성
  - 출력 생성: RPM, DEB, IPK 등의 패키지 피드, 이미지 파일, SDK를 생성
  - 오류 처리와 QA: 빌드 중 QA 테스트를 수행

--------

### 개발 환경

<img width="701" height="384" alt="개발환경" src="https://github.com/user-attachments/assets/56a3740a-4642-4342-ad33-37e1c4de2295" />


|항목                  |내용                              |
| ---------------------|----------------------------------|
| 보드                 | RaspBerryPi5                     |
| AI 가속기            | Hailo-8                          |
| 운영체제             | Ubuntu / Yocto 기반 Linux         |
| 언어                 | Python 3.8 / C                   |
| AI 프레임워크        | Pytorch / ONNX Runtime / Hailo SDK|
| 비전 프레임워크      | OpenCV / GStreamer                |
| 시뮬레이터           | Carla                             |

-------

### 하드웨어 구성

<img width="1247" height="553" alt="image" src="https://github.com/user-attachments/assets/83881d7c-a061-4597-8b1e-620f1ecab3da" />


------

### 설치

#### Ubuntu-RPi 크로스 컴파일 환경 구성

1. submodule 다운로드

```bash
$ git submodule update --init --recursive
```

2. 크로스 컴파일용 라이브러리 설치

  - ARM64 아키텍처 추가
  ```bash
  $ sudo dpkg --add-architecture arm64
  ```

  - /etc/apt/sources.list 파일 수정
  ```bash
  $ sudo vi /etc/apt/sources.list
  ```

  - ARM용 저장소 주소 추가
  ```bash
  # ARM64 아키텍처를 위한 Ubuntu Ports 저장소
  $ deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports jammy main restricted universe multiverse
  $ deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports jammy-updates main restricted universe multiverse
  $ deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports jammy-backports main restricted universe multiverse
  $ deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports jammy-security main restricted universe multiverse
  ```
   
  - ARM64용 패키지를 다운로드
  ```bash
  $ sudo apt update
  ```

  - ARM용 GStreamer 개발 라이브러리 설치
  ```bash
  $ sudo apt install -y libgstreamer1.0-dev:arm64 libgstreamer-plugins-base1.0-dev:arm64
  ```
---

#### 컴파일 방법

```bash
$ cd blackbox
```

- x86 빌드
```bash
$ make
```

- arm 빌드
```bash
$ make cross
```

- arm 빌드 및 타겟 보드로 전송(scp)
```bash
$ make deploy PI_USER=(USER_NAME) PI_IP=(USER_PASSWORD)
```

---
#### git clone

```bash
$ git clone https://github.com/~
$ cd ~
```

#### 의존성 설치
--------

- python3.8
- requirements.txt
- hailo SDK

---
- python3.8 의존성 설치
```bash
$ sudo apt update
$ sudo apt install -y build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev curl git libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

- pyenv 설치
```bash
$ curl https://pyenv.run | bash
```

- vi ~/.bashrc 변경
```bash
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
exec $SHELL
```

- python3.8 설치 & 적용
```bash
$ pyenv install 3.8.0
$ pyenv version
$ pyenv global 3.8.0
```

- requirements 설치
```bash
$ sudo apt update && sudo apt full-upgrade -y
$ sudo raspi-config  # Interface Options > PCIe > Enable
$ sudo reboot
```

```bash
#가상환경 접속 후
$ pip install -r requirements.txt
```

- hailo SDK 설치

```bash
$ sudo dpkg -i hailort_<version>_<architecture>.deb
$ sudo dpkg -i hailort-pcie-driver_<version>_all.deb
```

```bash
$ tar xzf hailo-rt-sdk-4.20.0-rpi.tar.gz
$ cd hailo-rt-sdk-4.20.0-rpi  # 또는 실제 디렉토리
$ ./install.sh
```

```bash
$ python3.8 -m venv hailo_env
$ source hailo_env/bin/activate
$ pip install hailort-4.20.0-cp38-cp38-linux_aarch64.whl
```

- SDK 설치 후 확인
```bash
$ dpkg -l | grep hailo
$ hailortcli fw-control identify
```

- 의존성 오류 시
```bash
$ sudo apt --fix-broken install
```

---
### 사용법

```bash
$ python vision_server.py
```

---
### 출력 구조

```bash
.
├── README.md
│
└── blackbox
    └── event6
  
```

---
### 결과 및 시연

```bash
추가 바람
```

---

### 팀원 소개

| 이름 | 담당 |
|------|------|
| **강송구** | PM / APP / HW |
| **김기환** | APP / Carla |
| **김민성** | Yocto |
| **이두현** | Gstream / AI / Carla |
| **정찬영** | AI / HW / 3D printing |




















