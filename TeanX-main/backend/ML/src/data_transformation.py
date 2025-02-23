import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Cleans input text by removing special characters, converting to lowercase,
    removing stopwords, and tokenizing the words.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\W', ' ', text).lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def transform_data(file_path, save_tokenizer_path, save_processed_path):
    """
    Loads dataset, cleans text, tokenizes, pads sequences, and saves processed data.
    
    Parameters:
    file_path (str): Path to the dataset.
    save_tokenizer_path (str): Path to save the tokenizer.
    save_processed_path (str): Path to save the processed dataset.
    """
    try:
        # Load dataset
        df = pd.read_csv(file_path)
        
        # Ensure necessary columns exist
        if 'text' not in df or 'label' not in df:
            raise ValueError("Dataset must contain 'text' and 'label' columns.")
        
        # Clean text
        df['text'] = df['text'].apply(clean_text)
        
        # Encode labels
        df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
        
        # Tokenization
        vocab_size = 9000
        tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
        tokenizer.fit_on_texts(X_train)
        
        # Convert text to sequences
        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_test_seq = tokenizer.texts_to_sequences(X_test)
        
        # Pad sequences
        max_length = 500
        X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
        X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding='post')
        
        # Save tokenizer
        with open(save_tokenizer_path, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save processed data
        processed_data = {
            'X_train': X_train_padded,
            'X_test': X_test_padded,
            'y_train': y_train.values,
            'y_test': y_test.values
        }
        with open(save_processed_path, 'wb') as handle:
            pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"✅ Data transformation complete. Processed data saved at: {save_processed_path}")
    except Exception as e:
        print(f"❌ Error in data transformation: {e}")

# Example usage
if __name__ == "__main__":
    file_path = "data/combined_news.csv"
    save_tokenizer_path = "models/tokenizer.pkl"
    save_processed_path = "data/processed_data.pkl"
    
    transform_data(file_path, save_tokenizer_path, save_processed_path)