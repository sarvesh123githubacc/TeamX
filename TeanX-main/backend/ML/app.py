import os
import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow requests from Chrome extension

# Paths
MODEL_PATH = r"C:\Users\sarve\Desktop\testing_repo\TeanX-main\backend\ML\bidirectional_lstm_model.h5"
TOKENIZER_PATH = r"C:\Users\sarve\Desktop\testing_repo\TeanX-main\backend\ML\notebooks\pickleFiles\models\tokenizer.pkl"

# Load Tokenizer
try:
    with open(TOKENIZER_PATH, "rb") as handle:
        tokenizer = pickle.load(handle)
    print("✅ Tokenizer Loaded Successfully")
except Exception as e:
    print(f"❌ Error loading tokenizer: {e}")

# Load Model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model Loaded Successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Function to preprocess text
def preprocess_text(text):
    """Tokenizes input text and pads it for model prediction."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=300)  # Adjust maxlen as per training
    return padded

# API Route to Analyze News
@app.route('/analyze', methods=['POST'])
def analyze_news():
    """Predicts news credibility and returns a message."""
    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Preprocess text
        input_data = preprocess_text(text)

        # Predict using ML model
        prediction = model.predict(input_data)[0][0]  # Assuming output is a probability

        # Convert probability to credibility score
        credibility_score = int(prediction * 100)  # Scale to 0-100

        # Determine credibility message
        if credibility_score >= 50:
            credibility_message = "Looks Real"
        else:
            credibility_message = "Looks False"

        return jsonify({
            "prediction": credibility_message,
            "credibility_score": credibility_score
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
