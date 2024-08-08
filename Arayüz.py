import tkinter as tk
from tkinter import font as tkfont
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
    signal, sample_rate = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=40)
    if mfccs.shape[1] > 40:
        mfccs = mfccs[:, :40]
    mfccs = np.mean(mfccs.T, axis=0)
    mfccs = np.expand_dims(mfccs, axis=0)
    predictions = model.predict(mfccs)
    predicted_label_index = np.argmax(predictions)
    predicted_label = classes[predicted_label_index]
    return predicted_label

def play_audio_response(response_text, audio_files):
    audio_file = audio_files.get(response_text, "default.mp3")
    if os.path.exists(audio_file):
        pygame.mixer.init()
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    else:
        print(f"Ses dosyası bulunamadı: {audio_file}")

# Arayüz
class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Yöresel Ağız Yapay Zekası")
        self.geometry("600x400")  # Boyut küçültüldü

        # Font ayarları
        self.header_font = tkfont.Font(family="Helvetica", size=14, weight="bold")
        self.button_font = tkfont.Font(family="Helvetica", size=12)

        # Başlık çerçevesi
        self.header_frame = tk.Frame(self, bg="#ADD8E6", pady=40)  
        self.header_frame.pack(fill=tk.X)

        # Başlık butonları
        tk.Button(self.header_frame, text="Sesli Konuşma", font=self.button_font, command=self.show_voice_page).pack(side=tk.LEFT, padx=40)
        tk.Button(self.header_frame, text="Yazarak Konuşma", font=self.button_font, command=self.show_text_page).pack(side=tk.LEFT, padx=40)
        tk.Button(self.header_frame, text="Sözlük", font=self.button_font, command=self.show_dictionary_page).pack(side=tk.LEFT, padx=40)

        # İçerik çerçevesi
        self.content_frame = tk.Frame(self, bg="#ADD8E6")  
        self.content_frame.pack(fill=tk.BOTH, expand=True)

        # JSON dosyalarını yükle
        self.kayseri_to_standard = load_json('Sorular.json')
        self.responses = load_json('Soru-Cevap.json')
        self.audio_files = load_json('Ses_Dosyası.json')

    def clear_frame(self):
        for widget in self.content_frame.winfo_children():
            widget.destroy()

    def show_voice_page(self):
        self.clear_frame()
        tk.Label(self.content_frame, text="Sesli Konuşma", font=self.header_font, bg="#ADD8E6").pack(pady=10)
        self.voice_output = tk.Text(self.content_frame, height=10, width=70, bg="#ffffff") 
        self.voice_output.pack(pady=10, padx=20)
        tk.Button(self.content_frame, text="Ses Kaydet ve Tanı", font=self.button_font, command=self.process_voice).pack(pady=10, side=tk.BOTTOM)

    def process_voice(self):
        file_path = record_audio()
        recognized_text = recognize_speech(file_path)
        self.voice_output.insert(tk.END, f"Tanımlanan Konuşma: {recognized_text}\n")
        standard_text = convert_shive_to_standard(recognized_text, self.kayseri_to_standard)
        self.voice_output.insert(tk.END, f"Standart Türkçe: {standard_text}\n")
        response_text = generate_response(standard_text, self.responses)
        self.voice_output.insert(tk.END, f"Cevap: {response_text}\n")
        play_audio_response(response_text, self.audio_files)

    def show_text_page(self):
        self.clear_frame()
        tk.Label(self.content_frame, text="Yazarak Konuşma", font=self.header_font, bg="#ADD8E6").pack(pady=10)
        self.text_input_frame = tk.Frame(self.content_frame, bg="#ADD8E6")
        self.text_input_frame.pack(pady=10, padx=20, fill=tk.X, side=tk.BOTTOM)
        self.text_input = tk.Entry(self.text_input_frame, font=self.button_font, width=40)  
        self.text_input.pack(side=tk.LEFT, padx=(0, 10), pady=10)
        tk.Button(self.text_input_frame, text="Gönder", font=self.button_font, command=self.process_text).pack(side=tk.RIGHT, pady=10)
        self.text_display = tk.Text(self.content_frame, height=10, width=70, bg="#ffffff")  
        self.text_display.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

    def process_text(self):
        kayseri_text = self.text_input.get()
        self.text_display.insert(tk.END, f"Şiveli Konuşma: {kayseri_text}\n")
        standard_text = convert_shive_to_standard(kayseri_text, self.kayseri_to_standard)
        self.text_display.insert(tk.END, f"Standart Türkçe: {standard_text}\n")
        response_text = generate_response(standard_text, self.responses)
        self.text_display.insert(tk.END, f"Cevap: {response_text}\n")
        self.text_input.delete(0, tk.END)

    def show_dictionary_page(self):
        self.clear_frame()
        tk.Label(self.content_frame, text="Sözlük", font=self.header_font, bg="#ADD8E6").pack(pady=10)
        self.text_input_frame = tk.Frame(self.content_frame, bg="#ADD8E6")
        self.text_input_frame.pack(pady=10, padx=20, fill=tk.X, side=tk.BOTTOM)
        self.text_input = tk.Entry(self.text_input_frame, font=self.button_font, width=40)  
        self.text_input.pack(side=tk.LEFT, padx=(0, 10), pady=10)
        tk.Button(self.text_input_frame, text="Ara", font=self.button_font, command=self.get_word_meaning).pack(side=tk.RIGHT, pady=10)
        self.dictionary_display = tk.Text(self.content_frame, height=10, width=70, bg="#ffffff")  
        self.dictionary_display.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

    def get_word_meaning(self):
        word = self.text_input.get()
        word_dict = load_json('Lehçe.json')
        meaning = word_dict.get(word, "Kelime bulunamadı")
        self.dictionary_display.insert(tk.END, f"{word}: {meaning}\n")
        self.text_input.delete(0, tk.END)

if __name__ == "__main__":
    app = Application()
    app.mainloop()
