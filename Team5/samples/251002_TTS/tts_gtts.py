from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import io

def speak_ko(text: str, slow: bool = False):
    tts = gTTS(text=text, lang='ko', slow=slow)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    audio = AudioSegment.from_file(buf, format="mp3")
    play(audio)

if __name__ == "__main__":
    speak_ko("안녕하세요. 지티티에스 한국어 테스트입니다. 파도와 부유물을 주의하세요.")


