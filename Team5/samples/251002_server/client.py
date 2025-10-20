import paho.mqtt.client as mqtt
import time

# MQTT 설정
BROKER = "10.10.14.73"  # 서버(라즈베리파이) IP
PORT = 1883
TOPIC = "project/vision"

client = mqtt.Client()
client.connect(BROKER, PORT, 60)

print("[INFO] Connected to broker, start publishing...")

# 테스트용 반복 발행
for i in range(5):
    msg = f"VISION@EVT@TRACK@id={i};conf=0.{i+5};zone=safe"
    client.publish(TOPIC, msg)
    print(f"[PUB] {TOPIC} → {msg}")
    time.sleep(1)

print("[DONE] Publishing finished")
client.disconnect()

