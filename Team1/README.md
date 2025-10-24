# [Intel 8기] 알파카

## About Project

> ###  <br /> 교통 및 주변 상황을 인식하는 스마트 시스템
>
> ### 개발기간: 2025.09.12 ~ 2025.10.22

## 📦주요 기능

<img src=https://github.com/user-attachments/assets/d3dfe0cb-f83b-430c-a839-fa51dcb1e784 width="500"/>
<img src=https://github.com/user-attachments/assets/047c481b-90bd-446f-99c7-6dd43247493c width="500"/>

<img src="https://github.com/user-attachments/assets/995aeaa4-8e05-4e69-aebe-60db998b305d" width="650"/>

### ⭐️ CCTV에서의 사고 차량 인식

### ⭐️ 차량 카메라에서 주변 상황 인식 및 위험 예측

### ⭐️ 클러스터를 통한 차량 제어 및 위험 경고

## 🤝 개발팀 소개

|                                      김성준                                        |                                      오정선                                       |                                      김경민                                      |
| :-------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------: | :------------------------------------------------------------------------------: |
| <img  width="100px" src="https://avatars.githubusercontent.com/u/147055391?v=4" /> | <img width="100px" src="https://avatars.githubusercontent.com/u/128763594?s=400&v=4" /> | <img width="100px" src="https://avatars.githubusercontent.com/u/233918499?v=4"/> |
|                    [@seolihan651jw](https://github.com/seolihan651)                     |                      [@gosumjigi](https://github.com/gosumjigi)                       |                      [@rudals6385-prog](https://github.com/rudals6385-prog)                     |
|                          [팀장]<br />프로젝트 총괄, ROS 구현,<br />데이터셋 제작, 형상 관리                          |                          [부팀장]<br />AI 모델링, ROS 구현,<br />웹 화면 구현                          |                                STM32 보드 차량<br />제작 및 배선,CAN 통신                                |


|                                          김영교                                         |                                      허진경                                      |
| :-------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------: |
|   <img width="100px" src="https://avatars.githubusercontent.com/u/221326759?v=4" />   |   <img width="100px" src="https://avatars.githubusercontent.com/u/228847706?v=4"/>   |
|                        [@mmc47047](https://github.com/mmc47047)                       |           [@Heo157](https://github.com/Heo157)           |
|         멀티캐스트 통신,<br />시스템 자동화          |            AGL 빌드            |

## 🖱 사용 기술

### 하드웨어 부품

- 라즈베리파이5 : <img src="https://img.shields.io/badge/Raspberry%20Pi-CC0000?style=flat&logo=Raspberry-Pi&logoColor=white" />
- STM32 2대 : <img src="https://img.shields.io/badge/STM32-03234B?style=flat&logo=STMicroelectronics&logoColor=white" />
  
### 소프트웨어

- 사용 언어:
<img src="https://img.shields.io/badge/C-00599C?style=flat-square&logo=C&logoColor=white" /> <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white" />

 - 프레임워크 및 라이브러리 :
<img src="https://img.shields.io/badge/ROS%202-F7B93E?style=flat&logo=ROS&logoColor=white" /> <img src="https://img.shields.io/badge/CARLA-3DDC84?style=flat&logo=Autonomous&logoColor=white" /> <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=OpenCV&logoColor=white" /> <img src="https://img.shields.io/badge/NumPy-013243?style=flat&logo=NumPy&logoColor=white" /> <img src="https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white" /> <img src="https://img.shields.io/badge/Flask-000000?style=flat&logo=Flask&logoColor=white" /> <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white" />

 - 통신 및 데이터 처리 : <img src="https://img.shields.io/badge/Docker-2496ED?style=flat&logo=Docker&logoColor=white" /> <img src="https://img.shields.io/badge/ROS%20Bridge-22314E?style=flat&logo=ROS&logoColor=white" />
   
[![Top Langs](https://github-readme-stats.vercel.app/api/top-langs/?username=gosumjigi)]([https://github.com/anuraghazra/github-readme-stats](https://github.com/kccistc/intel-08/new/main/Team1))

🔗 About Project (프로젝트 개요)
교통 및 주변 상황을 인식하는 스마트 시스템의 하부 제어 모듈입니다.

stm 차량은 ROS 2 (라즈베리 파이5) 시스템으로부터 UART 통신으로 명령을 받아, 차량의 모터(바퀴)를 정밀 제어하고, 제어 상태를 OLED 디스플레이에 시각화하여 운전자(또는 개발자)에게 정보를 제공합니다.


## ⚙️ 1. 사용된 주요 기술 및 하드웨어분류기술/모듈역할마이크로컨트롤러STM32F4 

### 전체 시스템 제어 및 통신 인터페이스상위 시스템ROS 2 (라즈베리 파이)주행 명령 및 속도 제어량 전송모터 
제어TIM2 (PWM)DC 모터 속도 제어 (주기: 9999)시각화SSD1306 (OLED)실시간 PWM 속도 계기판 출력통신 


## ⚡ 2. STM32 주요 기능 및 동작 원리
### 2.1. ROS 2 명령 수신 및 처리 📡(UART1)메인 루프에서 UART1을 통해 라즈베리 파이로부터 주행 명령을 수신합니다.프로토콜: "S,speed,direction\n" (예: S,50,0)속도 변환: 수신된 speed 값 (0~100)을 실제 모터 PWM 듀티 사이클
(0 ~ 9000, MAX_PWM_SPEED)로 변환하여 모터에 적용합니다. 

<img width="324" height="80" alt="image" src="https://github.com/user-attachments/assets/5c4142f7-1e67-4131-a918-467e51be55c4" />


## 📂 프로젝트 디렉토리 구조

```📦 project-root
├── docker-compose.yml                # 전체 컨테이너 구성 
├── README.md                         # 프로젝트 개요 문서
│
├── carla_simulator/                  # CARLA 시뮬레이터 관련 설정
│   ├── Dockerfile
│   ├── carla_config.py
│   ├── map/                          # 시뮬레이션 지도 및 환경 설정
│   └── scenario/                     # 주행 시나리오 스크립트
│
├── ros_ws/                           # ROS2 워크스페이스
│   ├── src/
│   │   ├── vehicle_control/          # 차량 제어 노드 (토픽/서비스 통신)
│   │   ├── sensor_interface/         # 센서 데이터 수집 노드
│   │   ├── ros_bridge/               # CARLA ↔ ROS 브리지 설정
│   │   └── msg/                      # 커스텀 메시지 정의
│   └── launch/                       # 실행용 launch 파일
│
├── raspberry_pi/                     # 라즈베리파이 측 코드
│   ├── main.py                       # 메인 제어 스크립트 (데이터 송수신)
│   ├── sensors/                      # 초음파, 카메라, IMU 등 센서 모듈
│   ├── comm/                         # gRPC/WebSocket 통신 모듈
│   └── utils/                        # 설정, 로그 등 공통 유틸
│
│
└── data/
    ├── logs/                         # 주행 기록, 센서 데이터 로그
    ├── images/                       # 수집된 이미지 데이터
    └── models/                       # 학습된 AI 모델 (필요 시)```
