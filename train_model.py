import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
import os
import pandas as pd
import cv2

# Etiketleri yükle
def load_labels():
    labels_df = pd.read_csv('dataset/labels.csv')
    return labels_df

# Veri setini yükle
def load_data():
    images = []
    labels = []
    data_dir = 'dataset/DATA'
    
    # Her bir sınıf klasörünü dolaş
    for class_folder in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_folder)
        if os.path.isdir(class_path):
            class_id = int(class_folder)  # Klasör adı sınıf ID'si
            
            # Her bir görüntüyü işle
            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                try:
                    # Görüntüyü yükle ve yeniden boyutlandır
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (32, 32))
                    img = img / 255.0  # Normalize et
                    
                    images.append(img)
                    labels.append(class_id)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
    
    return np.array(images), np.array(labels)

def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(58, activation='softmax')  # Sınıf sayısına göre ayarlandı
    ])
    
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

def train():
    # Etiketleri yükle
    labels_df = load_labels()
    print("Labels loaded. Total classes:", len(labels_df))
    
    # Veriyi yükle
    print("Loading data...")
    X, y = load_data()
    print("Data loaded. Total samples:", len(X))
    
    # Veriyi eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    
    # Modeli oluştur
    model = create_model()
    print("Model created.")
    
    # Modeli eğit
    history = model.fit(X_train, y_train, 
                       epochs=50,
                       batch_size=32,
                       validation_data=(X_test, y_test),
                       callbacks=[
                           tf.keras.callbacks.EarlyStopping(
                               monitor='val_loss',
                               patience=5,
                               restore_best_weights=True
                           )
                       ])
    
    # Modeli kaydet
    model.save('traffic_sign_model.h5')
    print("Model saved as traffic_sign_model.h5")
    
    # Eğitim sonuçlarını görselleştir
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    
    plt.savefig('training_results.png')
    print("Training results saved as training_results.png")

if __name__ == '__main__':
    train()
