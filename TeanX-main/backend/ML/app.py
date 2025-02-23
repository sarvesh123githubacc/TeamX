from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app)  # Enable CORS so your Chrome extension can fetch from this API

# Load the Keras model and the tokenizer
model = load_model('model.h5')
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Set the maximum sequence length (adjust this value to match your training configuration)
MAX_SEQUENCE_LENGTH = 100

@app.route('/predict', methods=['POST'])
def predict():
    # Expect JSON data with a "text" field
    data = request.get_json(force=True)
    text = data.get('text', None)
    
    if text is None:
        return jsonify({'error': 'No text provided.'}), 400

    # Convert the text to a sequence using the tokenizer
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    
    # Get the prediction from the model
    prediction = model.predict(padded_sequences)
    
    # Process the prediction as needed
    # Example: For binary classification, converting probability to class label
    predicted_class = int(prediction[0] > 0.5)

    return jsonify({
        'prediction': predicted_class,
        'raw_prediction': prediction[0].tolist()
    })

if __name__ == '__main__':
    # Set debug=False for production
    app.run(host='0.0.0.0', port=5000, debug=True)
