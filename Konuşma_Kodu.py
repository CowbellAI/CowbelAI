import numpy as np
import sounddevice as sd
import soundfile as sf
from tensorflow.keras.models import load_model
import librosa
import tempfile
import json
import re
import os
import pygame

# Model ve etiketleri yükleme
model = load_model('speech_recognition_model.h5')

# Etiketleri yükleme
with open('label_encoder.npy', 'rb') as f:
    classes = np.load(f, allow_pickle=True)

# JSON dosyasını okuyarak sözlüğü yükleme fonksiyonu
def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)

# Şive dönüştürme fonksiyonu
def convert_shive_to_standard(text, word_dict):
    text_cleaned = text.lower().strip()
    for key, value in word_dict.items():
        key_cleaned = key.lower().strip()
        if key_cleaned in text_cleaned:
            text_cleaned = re.sub(r'\b{}\b'.format(re.escape(key_cleaned)), value, text_cleaned, flags=re.IGNORECASE)
    return text_cleaned

# Cevap üretim fonksiyonu
def generate_response(standard_text, responses):
    standard_text = standard_text.lower().strip()
    for key, response in responses.items():
        key_cleaned = key.lower().strip()
        if key_cleaned in standard_text:
            return response
    return responses.get("default", "Bu konu hakkında ne söyleyeceğimi bilemiyorum.")

# Mikrofonla ses kaydetme
def record_audio(duration=5, samplerate=16000):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        filename = temp_file.name
        print("Kayıt başlıyor...")
        audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
        sd.wait()  # Kayıt tamamlanana kadar bekleyin
        sf.write(filename, audio_data, samplerate)
        print("Kayıt tamamlandı.")
    return filename

def recognize_speech(file_path):
    # Ses dosyasını yükleme
    signal, sample_rate = librosa.load(file_path, sr=None)
    
    # MFCC öznitelikleri çıkarma
    mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=40)
    
    # MFCC özniteliklerini pencereler halinde düzenleme (örneğin, ilk 40 pencere)
    if mfccs.shape[1] > 40:
        mfccs = mfccs[:, :40]  # Veya uygun pencereleri seçin

    # MFCC boyutunu (1, 40) formatına dönüştürme
    mfccs = np.mean(mfccs.T, axis=0)  # Ortalama alarak özetleme
    mfccs = np.expand_dims(mfccs, axis=0)
    
    # Tahmin yapma
    predictions = model.predict(mfccs)
    predicted_label_index = np.argmax(predictions)
    predicted_label = classes[predicted_label_index]
    
    return predicted_label

# Yanıt ses dosyasını oynatma
def play_audio_response(response_text, audio_files):
    audio_file = audio_files.get(response_text, "default.mp3")
    if os.path.exists(audio_file):
        pygame.mixer.init()
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():  # Bekle, müzik çalmaya devam ederken
            pygame.time.Clock().tick(10)
    else:
        print(f"Ses dosyası bulunamadı: {audio_file}")

# Tam uygulama
def main():
    # JSON dosyalarından kelimeleri, yanıtları ve ses dosyalarını yükle
    kayseri_to_standard = load_json('Sorular.json')
    responses = load_json('Soru-Cevap.json')
    audio_files = load_json('Ses_Dosyası.json')  # Ses dosyalarının JSON dosyasını yükleyin
    
    # Mikrofonla ses kaydetme
    file_path = record_audio()
    
    # Ses dosyasını tanıma
    recognized_text = recognize_speech(file_path)
    print("Tanımlanan Konuşma:", recognized_text)
    
    # Şiveyi standart Türkçeye çevirme
    standard_text = convert_shive_to_standard(recognized_text, kayseri_to_standard)
    print(f"Standart Türkçe: {standard_text}")

    # Standart Türkçeye çevrilen metne uygun cevap üretme
    response_text = generate_response(standard_text, responses)
    print(f"Cevap: {response_text}")

    # Yanıtı sesli çalma
    play_audio_response(response_text, audio_files)

if __name__ == "__main__":
    main()
