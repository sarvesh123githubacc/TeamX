import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
import pickle

# Load processed data
with open("data/processed_data.pkl", 'rb') as handle:
    processed_data = pickle.load(handle)

X_train, X_test = processed_data['X_train'], processed_data['X_test']
y_train, y_test = processed_data['y_train'], processed_data['y_test']

# Define model parameters
vocab_size = 9000
embedding_dim = 128
max_length = 500
lstm_units = 128
dropout_rate = 0.5

# Build the Bidirectional LSTM model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    Bidirectional(LSTM(lstm_units, return_sequences=True)),
    Dropout(dropout_rate),
    Bidirectional(LSTM(lstm_units)),
    Dropout(dropout_rate),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=64
)

# Save the trained model
model.save("models/bidirectional_lstm_model.h5")
print("✅ Model training complete. Model saved at: models/bidirectional_lstm_model.h5")
