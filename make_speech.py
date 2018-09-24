import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
import os

speech_file = open(os.getcwd() + '/all_speeches.txt')
data = speech_file.read().lower()[:1000]
speech_file.close()

characters = sorted(list(set(data)))
vocab_size = len(characters)
text_size = len(data)
char_to_index = {char: index for index, char in enumerate(characters)}
index_to_char = {index: char for index, char in enumerate(characters)}

# create time series pairs: sequence_length of chars (X) => next char (Y); (encoded as int)
# dataX[i] is an input paired with dataY[i], its ouput
sequence_length = 100
dataX = []
dataY = []

# one step/character at a time
for i in range(0, text_size - sequence_length, 1):
    input_sequence = data[i:i+sequence_length]
    output = data[i+sequence_length]
    dataX.append([char_to_index[char] for char in input_sequence])
    dataY.append(char_to_index[output])

#reshape dataX for Keras LSTM: [batch_size, time_stps, output_features]
X = np.reshape(dataX, (-1, sequence_length, 1))
#noramlization of input for LSTM
X = X / float(vocab_size)
#one-hot encoding of output
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
dataY = np.asarray(dataY).reshape(-1,1)
y = enc.fit_transform(dataY).toarray()

#simple RNN w/ LSTM layers
model = Sequential()
model.add(LSTM(256, input_shape = (X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

trump_speech = "we saw strangers shielding strangers from a hail of gunfire on the las vegas strip. we heard tales o"

#load trained model
file_name = "speech_weights-epoch:202-loss:0.0121.hdf5"
model.load_weights(os.getcwd() + "/" + file_name)
model.compile(loss='categorical_crossentropy', optimizer='adam')

int_speech = []
for char in trump_speech:
    int_speech.append(char_to_index[char])
x = np.array(int_speech)
x = np.reshape(x, (1,100,1))/float(vocab_size)
big_speech = int_speech.copy()

#generate next 500 characters
for i in range(800):
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    big_speech.append(index)
    int_speech = int_speech[1:]
    int_speech.append(index)
    x = np.array(int_speech)
    x = np.reshape(x, (1,100,1))/float(vocab_size)
with open("new_speech.txt", 'w') as ns:
    for int_char in big_speech:
        ns.write(index_to_char[int_char])
        print(index_to_char[int_char])
