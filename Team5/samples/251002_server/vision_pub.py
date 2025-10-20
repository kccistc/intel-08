import paho.mqtt.client as mqtt
import time, json

client = mqtt.Client()
client.connect("localhost", 1883, 60)

for i in range(5):
    event = {"src": "vision", "msg": f"person detected {i}"}
    client.publish("marine/vision", json.dumps(event))
    print("[VISION] Sent:", event)
    time.sleep(1)

