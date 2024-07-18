import os
from google.cloud import speech
from gtts import gTTS
import sounddevice as sd
import wavio
import numpy as np
from playsound import playsound

# Google Cloud API ayarları
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "crested-timer-428822-b2-695ed4c4e0b0.json"

# Ses kaydı parametreleri
samplerate = 16000  # Hertz
duration = 5  # seconds
filename = 'output.wav'

# Ses kaydetme fonksiyonu
def record_audio():
    print("Konuşmaya başlayabilirsiniz.")
    myrecording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()  # Wait until recording is finished
    wavio.write(filename, myrecording, samplerate, sampwidth=2)
    print("Kayıt tamamlandı.")
    return filename

# Google Speech-to-Text ile ses kaydını metne çevirme
def transcribe_audio(audio_path):
    client = speech.SpeechClient()
    with open(audio_path, "rb") as audio_file:
        content = audio_file.read()
    
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="tr-TR"
    )

    response = client.recognize(config=config, audio=audio)
    
    for result in response.results:
        return result.alternatives[0].transcript

# Yanıtı metin olarak oluşturma (basit bir örnek)
def generate_response(text):
    # Burada daha gelişmiş bir model (örn. GPT-3) kullanılabilir
    responses = {
        "Merhaba": "Merhaba! Nasılsınız?",
        "nasılsın": "İyiyim, teşekkürler. Siz nasılsınız?",
        "teşekkürler": "Rica ederim!"
    }
    return responses.get(text, "Anlayamadım, lütfen tekrar edin.")

# Metni sese çevirme ve çalma
def text_to_speech(text):
    tts = gTTS(text=text, lang='tr')
    tts.save("response.mp3")
    playsound("response.mp3")

# Tüm süreci çalıştırma
def main():
    audio_path = record_audio()
    text = transcribe_audio(audio_path)
    print("Siz: ", text)
    response = generate_response(text)
    print("Yanıt: ", response)
    text_to_speech(response)

if __name__ == "__main__":
    main()

