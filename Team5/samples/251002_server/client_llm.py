import paho.mqtt.client as mqtt
import openai  # OpenAI API 예시

BROKER = "10.10.14.73" 
PORT = 1883

TOPIC_REQ = "project/llm/req"
TOPIC_RESP = "project/llm/resp"

# OpenAI API 키 설정 (환경 변수로 두는 게 안전)
openai.api_key = "YOUR_API_KEY"

# 콜백: 메시지 수신
def on_message(client, userdata, msg):
    prompt = msg.payload.decode()
    print(f"[REQ] {prompt}")

    # LLM 호출 (간단 요약 예시)
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a maritime safety assistant."},
            {"role": "user", "content": f"Summarize this event log: {prompt}"}
        ]
    )

    summary = response["choices"][0]["message"]["content"].strip()
    print(f"[RESP] {summary}")

    # 응답을 다시 MQTT로 발행
    client.publish(TOPIC_RESP, summary)

# MQTT 클라이언트 준비
client = mqtt.Client()
client.on_message = on_message
client.connect(BROKER, PORT, 60)

# LLM 요청 구독
client.subscribe(TOPIC_REQ)

print("[LLM] Listening for requests...")
client.loop_forever()

