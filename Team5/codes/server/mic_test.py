import sounddevice as sd
from scipy.io.wavfile import write
from pydub import AudioSegment
import os

# ===== 설정 =====
SAMPLE_RATE = 16000  # Google STT 호환용 표준
DURATION = 5         # 녹음 시간 (초)
WAV_FILE = "test_record.wav"
MP3_FILE = "test_record.mp3"

print("🎙 녹음을 시작합니다. ({}초 동안 말하세요...)".format(DURATION))
audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
sd.wait()
print("✅ 녹음 완료!")

# WAV 파일로 저장
write(WAV_FILE, SAMPLE_RATE, audio)
print(f"💾 {WAV_FILE} 저장 완료")

# MP3로 변환
sound = AudioSegment.from_wav(WAV_FILE)
sound.export(MP3_FILE, format="mp3")
print(f"💾 {MP3_FILE} 저장 완료")

# ===== 재생 =====
print("🔊 녹음된 소리를 재생합니다...")
os.system(f"mpv --no-terminal {MP3_FILE}")

