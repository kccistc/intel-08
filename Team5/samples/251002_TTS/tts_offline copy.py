import pyttsx3

def speak_pyttsx3(text: str):
    engine = pyttsx3.init()

    # 속도 조절 (기본보다 빠르게 해서 아이 목소리 톤)
    engine.setProperty("rate", 210)  

    # 볼륨 (0.0 ~ 1.0)
    engine.setProperty("volume", 1.0)

    # 목소리 선택 (윈도우: female, Mac: com.apple.speech.synthesis.voice.yuna 등)
    voices = engine.getProperty("voices")
    for v in voices:
        if "female" in v.name.lower() or "yuna" in v.id.lower():
            engine.setProperty("voice", v.id)
            break

    # 문장마다 약간의 연결감을 주기 위해 이벤트 큐 사용
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    speak_pyttsx3("안녕하세요! 저는 아이처럼 상큼한 목소리를 내는 TTS입니다. 오늘도 즐겁게 지내요!")
