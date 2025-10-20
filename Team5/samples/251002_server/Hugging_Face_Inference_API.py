from huggingface_hub import InferenceClient

HF_TOKEN = "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"   # 네 토큰

client = InferenceClient(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct", 
    token=HF_TOKEN,
)

# 대화 형식으로 입력
messages = [
    {"role": "system", "content": "너는 항해 보조관이야."},
    {"role": "user", "content": "오늘 항해 로그: roll=-18, pitch=2, yaw=40. 상황을 자연어로 보고해줘."}
]

response = client.chat_completion(
    messages=messages,
    max_tokens=200,
    temperature=0.7,
)

print("=== LLaMA 응답 ===")
print(response.choices[0].message["content"])
