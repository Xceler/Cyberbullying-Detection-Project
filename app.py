import streamlit as st
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
import os

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load the saved model, tokenizer, and label encoder
model_path = "cyberbullying_detection.h5"
tokenizer_path = 'tokenizer.pickle'
label_encoder_path = 'label_encoder.pickle'

# Ensure files exist
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found.")
else:
    model = load_model(model_path)

if not os.path.exists(tokenizer_path):
    st.error(f"Tokenizer file '{tokenizer_path}' not found.")
else:
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)

if not os.path.exists(label_encoder_path):
    st.error(f"Label encoder file '{label_encoder_path}' not found.")
else:
    with open(label_encoder_path, 'rb') as handle:
        label_encoder = pickle.load(handle)

# Define preprocessing function
def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Removing Punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Removing Stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    # Combine tokens back into a string
    preprocessed_text = ' '.join(stemmed_tokens)
    return preprocessed_text

# Streamlit app
st.title("Cyberbullying Detection")

text_input = st.text_area("Enter a text to classify")

if st.button("Classify"):
    if text_input:
        # Check if model, tokenizer, and label encoder are loaded
        if 'model' not in globals():
            st.error("Model is not loaded.")
        elif 'tokenizer' not in globals():
            st.error("Tokenizer is not loaded.")
        elif 'label_encoder' not in globals():
            st.error("Label encoder is not loaded.")
        else:
            # Preprocess the input text
            preprocessed_text = preprocess_text(text_input)
            sequence = tokenizer.texts_to_sequences([preprocessed_text])
            sequence_pad = pad_sequences(sequence, maxlen=100)
            
            # Make prediction
            prediction = model.predict(sequence_pad)
            predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
            
            st.write(f"Prediction: {predicted_label[0]}")
    else:
        st.write("Please enter a text to classify.")
