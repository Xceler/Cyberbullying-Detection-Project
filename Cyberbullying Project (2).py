import pandas as pd 
import nltk 
import tensorflow as tf 
import zipfile 
import numpy as np 
import re             
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer 
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Embedding, LSTM, Dense 
from tensorflow.keras.callbacks import EarlyStopping 

#Extracting zipfile to get csv file for handling data 
zip_fil = 'cyberbullying-classification.zip'
with zipfile.ZipFile(zip_fil, 'r') as fil:
    fil.extractall() 


#Accessing CSV file 
fil = 'cyberbullying_tweets.csv'
data = pd.read_csv(fil)
print(data.head())

#Download nltk resources 
nltk.download('stopwords')
nltk.download('punkt')

#load your dataset
x = data['tweet_text'].values # Text data 
y = data['cyberbullying_type'].values #Categorical Data 

#Preprocessing function for data 
def preprocess_text(text):

    #Lowercasing 
    text = text.lower() 

    #Removing Punctuation 
    txt = re.sub(r'[^\w\s]', '', text)

    #Tokenization 
    tokens = word_tokenize(txt)

    #Removing stopwords 
    stop_words= set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    #Stemming 
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

    #combine tokens into a string 
    preprocessed_txt  = ' '.join(stemmed_tokens)
    return preprocessed_txt

#Apply preprocess function to x data 
x_preprocess = [preprocess_text(text) for text in x]

#Encode label for y data 
label_encode = LabelEncoder()
y_encoded = label_encode.fit_transform(y)

#Feature_extracion 
max_words = 10000
max_len = 100
tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(x_preprocess)
sequences = tokenizer.texts_to_sequences(x_preprocess)
x_pad = pad_sequences(sequences, maxlen = max_len)

#Split data into train and test data 
x_train, x_test, y_train, y_test = train_test_split(x_pad, y_encoded, test_size = 0.2, random_state = 42)

#Building LSTM (Long_Short_Term_Model)

embedding_dim = 100
lstm_units = 64
vocab_size = min(max_words, len(tokenizer.word_index) + 1)
num_classes = len(label_encode.classes_)

model = Sequential()
model.add(Embedding(input_dim = vocab_size, output_dim = embedding_dim, input_length =max_len))
model.add(LSTM(units = lstm_units))
model.add(Dense(units = num_classes, activation = 'softmax'))

model.compile(loss = 'sparse_categorical_crossentropy', 
              optimizer = 'adam', 
              metrics = ['accuracy'])
            

#Early Stopping for this model 
early_stopping = EarlyStopping(patience = 3, restore_best_weights = True) 
history= model.fit(x_train, y_train, epochs = 20,
                  batch_size = 64,
                  validation_data = (x_test, y_test),

                  callbacks = [early_stopping])


#Evaluate The model 
loss, accuracy = model.evaluate(x_test, y_test)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)

#Make predictions on new text data 
def predict_text(data):
    preprocessed_text = preprocess_text(data)
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    sequence_pad = pad_sequences(sequence, maxlen = max_len)
    prediction = model.predict(sequence_pad)
    predicted_label = label_encode.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

#Test predictions on new text data 
new_texts = [
    'You are ugly and nobody likes you',
    "What's your name"
]

for text in new_texts:
    prediction = predict_text(text)
    print(f'Text : {text}')
    print(f'Prediction: {prediction}')