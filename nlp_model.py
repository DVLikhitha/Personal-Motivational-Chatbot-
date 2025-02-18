import json
import numpy as np # linear algebra
import pandas as pd 
import re
import random
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, LayerNormalization, Dense, Dropout
import tensorflow 
import sklearn
# Load data from JSON file
with open(r"C:\Users\Arigala.Adarsh\Music\stream\intents.json", 'r') as f:
    data = json.load(f)

# Convert JSON data to DataFrame
df = pd.DataFrame(data['intents'])

# Create a dictionary to store patterns, tags, and responses
dic = {"tag":[], "patterns":[], "responses":[]}
for example in data['intents']:
    for pattern in example['patterns']:
        dic['patterns'].append(pattern)
        dic['tag'].append(example['tag'])
        dic['responses'].append(example['responses'])

# Convert dictionary to DataFrame
df = pd.DataFrame.from_dict(dic)


# Text preprocessing
tokenizer = Tokenizer(lower=True, split=' ')
tokenizer.fit_on_texts(df['patterns'])
vacab_size = len(tokenizer.word_index)
print(vacab_size )
# Convert text to sequences
ptrn2seq = tokenizer.texts_to_sequences(df['patterns'])
X = pad_sequences(ptrn2seq, padding='post')

# Encode labels
lbl_enc = LabelEncoder()
y = lbl_enc.fit_transform(df['tag'])

num_classes=len(df['tag'].unique())
# Define the model
model = Sequential()
model.add(Input(shape=(X.shape[1],)))
model.add(Embedding(input_dim= vacab_size + 1, output_dim=100))
model.add(LSTM(64, return_sequences=True))
model.add(LayerNormalization())
model.add(LSTM(64, return_sequences=True))
model.add(LayerNormalization())
model.add(LSTM(64, return_sequences=True))
model.add(LayerNormalization())
model.add(LSTM(64))
model.add(LayerNormalization())
model.add(Dense(256, activation="relu"))
model.add(LayerNormalization())
model.add(Dense(128, activation="relu"))
model.add(LayerNormalization())
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation="softmax"))
model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
# Model summary
model.summary()

# Train the model
model_history = model.fit(x=X,
                           y=y,
                           batch_size=10,
                           callbacks=[tensorflow.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)],
                           epochs=100)

 #Function to generate response
def generate_answer(user_input): 
    while True:
        pattern = user_input  
        
        if pattern.lower() == 'quit':
            break

        text = []
        txt = re.sub('[^a-zA-Z\']', ' ', pattern)
        txt = txt.lower()
        txt = txt.split()
        txt = " ".join(txt)
        text.append(txt)
        

        x_test = tokenizer.texts_to_sequences(text)
        x_test = pad_sequences(x_test, padding='post', maxlen=X.shape[1])
        
        y_pred = model.predict(x_test)
        y_pred = y_pred.argmax()
        
        tag = lbl_enc.inverse_transform([y_pred])[0]
        responses = df[df['tag'] == tag]['responses'].values[0]
        print("Model: {}".format(random.choice(responses)))
        return random.choice(responses)
 