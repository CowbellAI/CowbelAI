import os
import numpy as np
import librosa
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, f1_score
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import History
from itertools import cycle

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
    history = model.fit(audio_data, y_encoded, epochs=50, batch_size=16, validation_split=0.2)

    # Modeli kaydetme
    print(f"Saving model")
    model.save('speech_recognition_model.h5')

    # Etiketleri kaydetme
    with open('label_encoder.npy', 'wb') as f:
        np.save(f, label_encoder.classes_)

    print("Model and labels saved successfully.")

    # Eğitim metriklerini değerlendirme
    print(f"Evaluating model")
    y_pred = model.predict(audio_data)
    y_pred_labels = np.argmax(y_pred, axis=1)

    # Classification report (Doğruluk, Kesinlik, Hassaslık, F1 Skoru)
    print(f"Classification report:")
    report = classification_report(y_encoded, y_pred_labels, target_names=label_encoder.classes_, zero_division=0)
    print(report)
    
    # Confusion matrix
    print(f"Confusion Matrix:")
    conf_matrix = confusion_matrix(y_encoded, y_pred_labels)
    print(conf_matrix)
    
    # Çok sınıflı AUC-ROC Eğrisi
    print(f"Plotting AUC-ROC curve for multiclass classification")
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(label_encoder.classes_)):
        fpr[i], tpr[i], _ = roc_curve(y_encoded, y_pred[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    plt.figure()
    for i, color in zip(range(len(label_encoder.classes_)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(label_encoder.classes_[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for multiclass')
    plt.legend(loc="lower right")
    plt.show()

    # Kaybın Değişimi
    print(f"Plotting loss over epochs")
    plt.figure()
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # Hiperparametrelerin Değeri ve Etkisi
    print(f"Hyperparameters:")
    print(f"Learning rate: {model.optimizer.learning_rate.numpy()}")
    print(f"Batch size: {16}")
    print(f"Epochs: {50}")
    print(f"Model evaluation metrics:")
    print(f"Final accuracy: {history.history['accuracy'][-1]}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]}")

except ValueError as ve:
    print(f"ValueError: {ve}")

except Exception as e:
    print(f"An error occurred: {e}")
