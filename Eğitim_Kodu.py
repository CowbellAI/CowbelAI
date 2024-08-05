import os
import numpy as np
import librosa
import soundfile as sf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
import json

# Ses dosyaları ve transkriptlerin bulunduğu dizinler
audio_dir = 'audio_files/'
transcript_path = 'transcripts.json'

# Ses dosyalarını ve transkriptleri okuma
with open(transcript_path, 'r', encoding='utf-8') as f:
    transcripts = json.load(f)

# Ses dosyalarını ve etiketleri işleme
audio_data = []
labels = []
target_length = 40  # MFCC özniteliklerinin sütun sayısı

for file_name, transcript in transcripts.items():
    file_path = os.path.join(audio_dir, file_name)
    
    try:
        # Ses dosyasını yükleme
        print(f"Processing file: {file_path}")
        signal, sample_rate = librosa.load(file_path, sr=None)
        
        # Ses sinyalinin uzunluğunu kontrol et
        if len(signal) < 2:
            print(f"File {file_path} is too short.")
            continue
        
        # MFCC öznitelikleri çıkarma
        mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=target_length)
        
        # MFCC özniteliklerinin her zaman aynı uzunlukta olduğundan emin olun
        if mfccs.shape[1] < target_length:
            print(f"Padding MFCCs for file: {file_path}")
            pad_width = target_length - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
        elif mfccs.shape[1] > target_length:
            print(f"Trimming MFCCs for file: {file_path}")
            mfccs = mfccs[:, :target_length]
        
        # Ortalama alma (her ses parçası için öznitelik vektörüne dönüştürme)
        mfccs_mean = np.mean(mfccs, axis=1)
        
        # Audio data ve labels listesine ekleme
        audio_data.append(mfccs_mean)
        labels.append(transcript)
        
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        continue

# Numpy array'e dönüştürme
try:
    print(f"Converting audio data to numpy array")
    audio_data = np.array(audio_data)  # MFCC özniteliklerini numpy array'e dönüştürme
    labels = np.array(labels)  # Etiketleri numpy array'e dönüştürme
    
    # Etiketlerin boyutunu kontrol et
    print(f"Audio data shape: {audio_data.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Label encoding
    print(f"Label encoding")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    
    # Etiketlerin doğruluğunu kontrol et
    print(f"Classes in label encoder: {label_encoder.classes_}")
    
    # Model oluşturma
    print(f"Building model")
    model = models.Sequential([
        layers.Input(shape=(audio_data.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(label_encoder.classes_), activation='softmax')
    ])

    # Modeli derleme
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Modeli eğitme
    print(f"Training model")
    model.fit(audio_data, y_encoded, epochs=50, batch_size=16, validation_split=0.2)

    # Modeli kaydetme
    print(f"Saving model")
    model.save('speech_recognition_model.h5')

    # Etiketleri kaydetme
    with open('label_encoder.npy', 'wb') as f:
        np.save(f, label_encoder.classes_)

    print("Model and labels saved successfully.")

except ValueError as ve:
    print(f"ValueError: {ve}")

except Exception as e:
    print(f"An error occurred: {e}")
