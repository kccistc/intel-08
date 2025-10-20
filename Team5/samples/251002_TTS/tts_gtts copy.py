from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import io

def change_pitch(sound: AudioSegment, semitones: float) -> AudioSegment:
    """세미톤 단위로 피치 조절"""
    new_sample_rate = int(sound.frame_rate * (2.0 ** (semitones / 12.0)))
    pitched = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate})
    return pitched.set_frame_rate(sound.frame_rate)

def speak_gtts(text: str, pitch_semitones: float = 5):
    tts = gTTS(text=text, lang="ko")
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    audio = AudioSegment.from_file(buf, format="mp3")
    audio = change_pitch(audio, pitch_semitones)  # 아이 목소리 느낌
    play(audio)

if __name__ == "__main__":
    speak_gtts("안녕하세요! 저는 어린이 목소리 톤의 TTS에요. 오늘 기분이 너무 좋아요!", pitch_semitones=6)
