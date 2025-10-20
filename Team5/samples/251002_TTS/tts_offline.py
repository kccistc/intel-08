import pyttsx3

def list_voices(engine):
    for i, v in enumerate(engine.getProperty('voices')):
        print(i, v.id, v.languages if hasattr(v, "languages") else "")

def tts(text: str, rate: int = 180, volume: float = 1.0, voice_hint: str = "ko"):
    engine = pyttsx3.init()  # espeak/NS/SAPI 자동 선택
    engine.setProperty('rate', rate)
    engine.setProperty('volume', volume)

    # 한국어 보이스 선택 (아이디/언어 코드에 'ko'가 포함된 보이스 탐색)
    voice_id = None
    for v in engine.getProperty('voices'):
        lang_tags = ",".join(getattr(v, "languages", []) or [])
        if ("ko" in v.id.lower()) or ("korean" in v.name.lower()) or ("ko" in lang_tags.lower()):
            voice_id = v.id
            break
    if voice_id:
        engine.setProperty('voice', voice_id)
    else:
        print("[warn] 한국어 보이스를 찾지 못했어요. 기본 보이스로 진행합니다. (espeak-ng ko 설치 권장)")

    engine.say(text)
    engine.runAndWait()
# hi!
if __name__ == "__main__":
    sample = "안녕하세요. 오프라인 TTS 테스트입니다. 안전 항해를 위해 주의하세요."
    tts(sample)


