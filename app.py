import pandas as pd
import numpy as np
import re
import nltk
import pickle
import tensorflow as tf
from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('punkt_tab')
nltk.download('stopwords')

app = Flask(__name__)

model = tf.keras.models.load_model('model_learning_style.h5')

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)


with open('max_len.pkl', 'rb') as f:
    max_len = pickle.load(f)


stop_words = set(stopwords.words('indonesian'))

def lower_clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]','',text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def preprocess_text(texts, tokenizer, max_len):

    cleaned_texts = [lower_clean(text) for text in texts]


    texts_seq = tokenizer.texts_to_sequences(cleaned_texts)
    texts_padded = pad_sequences(texts_seq, maxlen=max_len, padding='post')

    return texts_padded

@app.route('/')
def index():
    return jsonify({"message": "Api deteksi learning style"})
    
@app.route('/predict2', methods=['POST'])
def predict_learning_style():
    try:
  
        data = request.json
        texts = data.get('texts', [])

        if not texts:
            return jsonify({"error": "No text provided"}), 400

  
        preprocessed_texts = preprocess_text(texts, tokenizer, max_len)

  
        predictions = model.predict(preprocessed_texts)
   
        predicted_labels = [label_encoder.inverse_transform([pred.argmax()])[0] for pred in predictions]

        response = []
        for text, pred, label in zip(texts, predictions, predicted_labels):
            response.append({
                "text": text,
                "learning_style": label,
                "probabilities": {
                    "Visual": float(pred[0]),
                    "Auditory": float(pred[1]),
                    "Kinesthetic": float(pred[2])
                }
            })

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3133)