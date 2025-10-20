import paho.mqtt.client as mqtt
import smbus2
import time
import math
import sys
import json 
from datetime import datetime, timezone

# ====================================================
# 1. 환경 설정 및 상수 정의
# ====================================================

# MQTT 설정
BROKER = "10.10.14.73" 
PORT = 1883

# 🚨🚨 IMU_USER 인증 정보 추가 🚨🚨
MQTT_USERNAME = "IMU_USER"      # 등록된 IMU 사용자 이름
MQTT_PASSWORD = "sksk"  # 등록된 IMU 사용자 비밀번호

# 수정: 모듈 이름 및 토픽 분리
IMU_MODULE = "IMU"
RAW_TOPIC = "project/imu/RAW"  # 원시 데이터 (정상/INFO 레벨)
ALERT_TOPIC = "project/imu/ALERT" # 경고/위험 데이터 (CRITICAL/WARNING 레벨)

# 🚨 새로운 상수: 위험 임계치
CRITICAL_ROLL_THRESHOLD = 30.0 # Roll 각도 ±30도를 전복 위험 CRITICAL 임계치로 설정 (선박 환경에 맞게 조정 필요)

# MPU-6050 I2C 주소 및 레지스터 주소
MPU6050_ADDR = 0x68
I2C_BUS = 1

# 레지스터 주소 및 스케일 상수
PWR_MGMT_1 = 0x6B
GYRO_CONFIG = 0x1B
ACCEL_XOUT_H = 0x3B 
ACCEL_YOUT_H = 0x3D
ACCEL_ZOUT_H = 0x3F
GYRO_XOUT_H = 0x43
GYRO_YOUT_H = 0x45 
GYRO_ZOUT_H = 0x47

# 스케일 상수 및 필터 계수
ACCEL_SCALE = 16384.0 # 가속도계 1g 스케일 (±2g 기준)
GYRO_SCALE = 131.0 # 자이로스코프 스케일 (±250 deg/s 기준)
RAD_TO_DEG = 57.2957795 # 라디안을 도로 변환
ALPHA = 0.98 # 상보 필터 계수 (자이로스코프 신뢰도 98%)

# ====================================================
# 2. 전역 변수 및 I2C 통신 함수
# ====================================================

bus = None
# 필터링된 Roll/Pitch 각도 및 Yaw 통합 각도
filtered_roll_angle = 0.0
filtered_pitch_angle = 0.0
integrated_yaw_angle = 0.0

# 자이로 오프셋 저장
Gx_offset = 0.0
Gy_offset = 0.0
Gz_offset = 0.0

def now_str():
    """ISO 8601 형식의 현재 UTC 시각을 반환합니다."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# read_word_2c, mpu6050_init 함수는 기존과 동일하게 유지됩니다.
def read_word_2c(reg):
    """MPU-6050에서 16비트 데이터를 읽고 2의 보수 처리합니다."""
    global bus
    try:
        high = bus.read_byte_data(MPU6050_ADDR, reg)
        low = bus.read_byte_data(MPU6050_ADDR, reg + 1)
    except IOError:
        print(f"[{now_str()}] ERROR I2C :: 데이터 읽기 실패. 센서 연결 확인 필요.")
        # 데이터 읽기 실패 시 복구를 시도하거나, 프로그램 종료가 안전할 수 있습니다.
        raise # 예외를 발생시켜 메인 루프에서 처리하도록 함
        
    val = (high << 8) + low
    if val >= 0x8000:
        return -((65535 - val) + 1)
    else:
        return val

def mpu6050_init():
    """MPU-6050 센서 초기화 및 I2C 버스 열기"""
    global bus
    try:
        bus = smbus2.SMBus(I2C_BUS)
        bus.write_byte_data(MPU6050_ADDR, PWR_MGMT_1, 0x00) # 슬립 모드 해제
        bus.write_byte_data(MPU6050_ADDR, GYRO_CONFIG, 0x00) # 자이로 스케일 ±250 deg/s
        bus.write_byte_data(MPU6050_ADDR, 0x1C, 0x00) # 가속도계 스케일 ±2g 설정
        print(f"[{now_str()}] INFO Sensor :: MPU-6050 Initialized successfully.")
        time.sleep(0.1)
    except FileNotFoundError:
        print(f"[{now_str()}] ❌ CRITICAL: I2C 버스 파일 ({I2C_BUS})을 찾을 수 없습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"[{now_str()}] ❌ CRITICAL: MPU-6050 초기화 오류: {e}")
        sys.exit(1)

# ====================================================
# 3. 각도 계산 및 필터링 함수
# ====================================================

def accel_roll(ay, az):
    """가속도계 데이터로 Roll 각도 계산 (X축 회전)"""
    return math.atan2(ay, az) * RAD_TO_DEG

def accel_pitch(ax, ay, az):
    """가속도계 데이터로 Pitch 각도 계산 (Y축 회전)"""
    # X축 가속도와 YZ 평면의 벡터 합 간의 각도를 계산하여 더 안정적인 Pitch 값을 얻습니다.
    return math.atan2(-ax, math.sqrt(ay*ay + az*az)) * RAD_TO_DEG

def complementary_filter(accel_angle, gyro_rate, prev_angle, dt):
    """상보 필터를 적용하여 Roll 또는 Pitch 각도를 계산합니다."""

    # 자이로스코프 값 적분 (이전 각도 + 자이로 변화량)
    gyro_angle = prev_angle + gyro_rate * dt

    # 상보 필터 적용: 자이로(단기) 98% + 가속도(장기) 2%
    filtered_angle = ALPHA * gyro_angle + (1.0 - ALPHA) * accel_angle

    return filtered_angle

# ====================================================
# 4. 메인 실행 함수 (통합 로직)
# ====================================================

def main():
    global Gx_offset, Gy_offset, Gz_offset
    global filtered_roll_angle, filtered_pitch_angle, integrated_yaw_angle

    # 1. 센서 초기화
    mpu6050_init()

    # 2. MQTT 클라이언트 생성 및 연결
    client = mqtt.Client(client_id="IMU_Client", protocol=mqtt.MQTTv311)

    # 사용자 이름 및 비밀번호 설정
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

    try:
        client.connect(BROKER, PORT, 60)
        client.loop_start() 
        print(f"[{now_str()}] INFO MQTT :: Client connected to {BROKER}:{PORT}")
    except Exception as e:
        print(f"[{now_str()}] ❌ CRITICAL: MQTT Connection failed: {e}")
        sys.exit(1)
    
    # 3. Gyro 3축 오프셋 캘리브레이션
    print(f"[{now_str()}] INFO Sensor :: Calibrating Gyro...")
    sum_Gx_raw, sum_Gy_raw, sum_Gz_raw = 0, 0, 0
    calibration_count = 100
    for _ in range(calibration_count):
        try:
            sum_Gx_raw += read_word_2c(GYRO_XOUT_H)
            sum_Gy_raw += read_word_2c(GYRO_YOUT_H)
            sum_Gz_raw += read_word_2c(GYRO_ZOUT_H)
        except:
            time.sleep(0.01)
            continue
        time.sleep(0.01)
        
    Gx_offset = (sum_Gx_raw / calibration_count) / GYRO_SCALE
    Gy_offset = (sum_Gy_raw / calibration_count) / GYRO_SCALE
    Gz_offset = (sum_Gz_raw / calibration_count) / GYRO_SCALE
    print(f"[{now_str()}] INFO Sensor :: Gyro Offsets (X/Y/Z): {Gx_offset:.2f}/{Gy_offset:.2f}/{Gz_offset:.2f} deg/s")

    # 4. 초기 Roll/Pitch 각도 설정
    # 초기 가속도계 데이터 읽기
    try:
        AccX_init = read_word_2c(ACCEL_XOUT_H)
        AccY_init = read_word_2c(ACCEL_YOUT_H)
        AccZ_init = read_word_2c(ACCEL_ZOUT_H)
    except:
        print(f"[{now_str()}] ❌ CRITICAL: Initial sensor read failed.")
        sys.exit(1)
        
    Ax_init, Ay_init, Az_init = AccX_init / ACCEL_SCALE, AccY_init / ACCEL_SCALE, AccZ_init / ACCEL_SCALE

    filtered_roll_angle = accel_roll(Ay_init, Az_init)
    filtered_pitch_angle = accel_pitch(Ax_init, Ay_init, Az_init)
    # Yaw는 초기화 시 0으로 설정
    integrated_yaw_angle = 0.0 
    print(f"[{now_str()}] INFO Sensor :: Initial Angles (R/P/Y): {filtered_roll_angle:.2f} / {filtered_pitch_angle:.2f} / {integrated_yaw_angle:.2f} deg")

    # 5. 메인 측정 및 발행 루프
    last_time_s = time.time() 

    try:
        while True:
            current_time_s = time.time()
            dt = current_time_s - last_time_s
            last_time_s = current_time_s

            # --- 5-1. 센서 데이터 읽기 ---
            try:
                AccX = read_word_2c(ACCEL_XOUT_H)
                AccY = read_word_2c(ACCEL_YOUT_H)
                AccZ = read_word_2c(ACCEL_ZOUT_H)
                GyroX = read_word_2c(GYRO_XOUT_H)
                GyroY = read_word_2c(GYRO_YOUT_H)
                GyroZ = read_word_2c(GYRO_ZOUT_H)
            except Exception as e:
                print(f"[{now_str()}] ⚠️ WARNING: I2C read error. Skipping frame. ({e})")
                time.sleep(0.01)
                continue
            
            # --- 5-2. 스케일 및 오프셋 적용 ---
            Ax, Ay, Az = AccX / ACCEL_SCALE, AccY / ACCEL_SCALE, AccZ / ACCEL_SCALE
            Gx = GyroX / GYRO_SCALE - Gx_offset # Roll Rate (X)
            Gy = GyroY / GYRO_SCALE - Gy_offset # Pitch Rate (Y)
            Gz = GyroZ / GYRO_SCALE - Gz_offset # Yaw Rate (Z)

            # --- 5-3. 각도 계산 및 필터링 ---
            # Roll (X축 회전)
            accel_roll_angle = accel_roll(Ay, Az)
            filtered_roll_angle = complementary_filter(accel_roll_angle, Gx, filtered_roll_angle, dt)

            # Pitch (Y축 회전)
            accel_pitch_angle = accel_pitch(Ax, Ay, Az)
            filtered_pitch_angle = complementary_filter(accel_pitch_angle, Gy, filtered_pitch_angle, dt)

            # Yaw (Z축 회전) - 자이로스코프 적분만 사용 (오차 누적 주의!)
            integrated_yaw_angle += Gz * dt 

            # Yaw 각도를 0~360 범위로 유지
            integrated_yaw_angle = integrated_yaw_angle % 360.0
            if integrated_yaw_angle < 0:
                integrated_yaw_angle += 360.0

            # --- 5-4. 🚨 CRITICAL 알람 체크 (우선순위) ---
            roll_abs = abs(filtered_roll_angle)
            if roll_abs > CRITICAL_ROLL_THRESHOLD:
                alert_message = f"🚨 긴급 전복 위험! Roll 각도 {roll_abs:.2f}° 초과 ({CRITICAL_ROLL_THRESHOLD:.2f}°)."
                alert_payload = json.dumps({
                    "timestamp": now_str(),
                    "module": IMU_MODULE,
                    "level": "CRITICAL", # 🚨 CRITICAL 레벨 적용
                    "message": alert_message,
                    "details": {"roll_angle": round(filtered_roll_angle, 2), "threshold": CRITICAL_ROLL_THRESHOLD}
                })
                # 기존 알람을 무시하고 즉시 발행 (QoS 1)
                client.publish(ALERT_TOPIC, alert_payload, qos=1) 
                print(f"[{now_str()}] 🚨🚨 CRITICAL PUB :: {ALERT_TOPIC} → {alert_message}")
            
            # --- 5-5. RAW 데이터 발행 (INFO 레벨) ---
            raw_payload = json.dumps({
                "roll": round(filtered_roll_angle, 2),
                "pitch": round(filtered_pitch_angle, 2),
                "yaw": round(integrated_yaw_angle, 2),
                "dt": round(dt, 4),
                "level": "INFO", # 🚨 INFO 레벨 추가
                "module": IMU_MODULE, # 모듈 이름 추가
                "timestamp": now_str()
            })
            
            # --- 5-5. 발행 ---
            result, mid = client.publish(RAW_TOPIC, raw_payload, qos=0)
            
            if result == mqtt.MQTT_ERR_SUCCESS:
                print(f"[{now_str()}] INFO PUB :: {RAW_TOPIC} → R:{filtered_roll_angle:6.2f} P:{filtered_pitch_angle:6.2f} Y:{integrated_yaw_angle:6.2f} deg | dt: {dt:.4f}s")
            # else: (RAW 발행 실패는 로그만 남기고 무시 가능)

            # 6. 다음 루프까지 대기 (1000ms 대기 -> 약 1 FPS)
            time.sleep(1) 
            
    except KeyboardInterrupt:
        print(f"\n[{now_str()}] INFO System :: Measurement stopped by user.")
    except Exception as e:
        print(f"\n[{now_str()}] ❌ ERROR System :: An unexpected error occurred: {e}")
    finally:
        client.loop_stop()
        client.disconnect() 
        print(f"[{now_str()}] INFO MQTT :: Client disconnected.")
        if bus:
            bus.close()
        sys.exit(0)

if __name__ == "__main__":
    main()
