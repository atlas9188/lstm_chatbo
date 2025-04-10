import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

# Define the path to the model and tokenizer
MODEL_PATH = 'model.keras'
TOKENIZER_PATH = 'tokenizer.pkl'

# Load the model
@st.cache(allow_output_mutation=True)
def load_keras_model():
    model = load_model(MODEL_PATH)
    return model

model = load_keras_model()

# Load the tokenizer
@st.cache(allow_output_mutation=True)
def load_tokenizer():
    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

tokenizer = load_tokenizer()

# Maximum sequence length: adjust based on the training details of the model
MAX_SEQUENCE_LENGTH = 20

def preprocess_input(user_input):
    # Convert input text to a sequence of integers
    sequence = tokenizer.texts_to_sequences([user_input])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    return padded_sequence

def generate_response(input_text):
    processed_input = preprocess_input(input_text)
    prediction = model.predict(processed_input)
    # Assuming the model returns indices; adjust as necessary based on model output
    response_index = np.argmax(prediction, axis=1)[0]
    response_word = tokenizer.index_word.get(response_index, "Sorry, I didn't understand that.")
    return response_word

# Streamlit interface
st.title('Chatbot Interface')
user_input = st.text_input("Type your message here:")

if st.button('Send'):
    if user_input:
        response = generate_response(user_input)
        st.text_area("Chatbot says:", value=response, height=100, max_chars=None)
    else:
        st.error("Please enter some text to chat.")
