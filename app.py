from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import pandas as pd

app = Flask(__name__)

# Model yükleme
model = tf.keras.models.load_model('traffic_sign_model.h5')

# Trafik işareti sınıflarının isimleri
labels_df = pd.read_csv('dataset/labels.csv')
CLASSES = dict(zip(labels_df['ClassId'], labels_df['Name']))

def preprocess_image(image):
    # Resmi 32x32 boyutuna getir
    image = image.resize((32, 32))
    # Numpy dizisine çevir
    img_array = np.array(image)
    # Normalize et
    img_array = img_array / 255.0
    # Batch boyutunu ekle
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya yüklenmedi'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'})
    
    try:
        # Resmi yükle ve ön işleme yap
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        processed_image = preprocess_image(image)
        
        # Tahmin yap
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction[0])
        
        # Sonucu döndür
        result = {
            'class': CLASSES.get(predicted_class, 'Bilinmeyen işaret'),
            'confidence': float(prediction[0][predicted_class])
        }
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
