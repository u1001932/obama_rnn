import os
import numpy as np

speech_file = open(os.getcwd() + '/all_speeches.txt')
data = speech_file.read().lower()[:1000]
speech_file.close()


characters = sorted(list(set(data)))
vocab_size = len(characters)
text_size = len(data)
char_to_index = {char: index for index, char in enumerate(characters)}
index_to_char = {index: char for index, char in enumerate(characters)}


import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint

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
print("done")
#reshape dataX for Keras LSTM: [batch_size, time_stps, output_features]
X = numpy.reshape(dataX, (-1, sequence_length, 1))
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

#checkpoints
checkpoint_path = "speech_weights-epoch:{epoch:02d}-loss:{loss:.4f}.hdf5"
#saves model (with decreased loss) 
checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks = [checkpoint]

#fit (callbacks includes checkpount func, saving model at each iteration)
model.fit(X, y, epochs=500, batch_size=64, callbacks=callbacks)