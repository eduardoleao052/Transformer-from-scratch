import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import LSTM, Dropout, Dense
# from keras.optimizers import Adam
# from keras.callbacks import ModelCheckpoint
# from sklearn.model_selection import train_test_split

import numpy 
# [load data from file]
raw_text = open('C:/Users/twich/OneDrive/Documentos/NeuralNets/rnn/data/way_of_kings.txt', 'r', encoding = 'utf8').read().lower()
# create mapping of unique chars to integers
chars = sorted(list(set(raw_text + '\b')))
char_to_int = dict((c, i) for i, c in enumerate(chars)) 
seq_length = 100
dataX = []
dataY = []
for i in range(0, len(chars) - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out]) 

# reshape X to be [samples, time steps, features]
#X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
#X = X / float(n_vocab)
# one hot encode the output variable
#y = np_utils.to_categorical(dataY)

# define the LSTM model
model = tf.keras.Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), 
return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# define the checkpoint
filepath="models/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, 
save_best_only=False, mode='min')
callbacks_list = [checkpoint]

# fix random seed for reproducibility
seed = 8
numpy.random.seed(seed)
# split into 80% for train and 20% for test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
  random_state=seed)

# train the model
model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=18, 
  batch_size=256, callbacks=callbacks_list, verbose=1)

#=================================================================================#


filename = "weights-improvement-18-1.5283.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')
int_to_char = dict((i, c) for i, c in enumerate(chars))
# pick a random seed
start = numpy.random.randint(0, len(dataX)-1)
print("Seed:")
print("\"", ''.join([int_to_char[value] for value in X_train[start:start+10]]), "\"")
pattern = X_train[start+10]
# generate characters
for i in range(1000):
  x = numpy.reshape(pattern, (1, len(pattern), 1))
  x = (x / float(x.shape[1])) + (numpy.random.rand(1, len(pattern), 1) * 0.01)
  prediction = model.predict(x, verbose=0)
  idx = numpy.random.choice(y.shape[1], 1, p=prediction[0])[0]#print(index)
  result = int_to_char[idx]
  print(result,end='')
  pattern.append(idx)
  pattern = pattern[1:len(pattern)]
print("\nDone.") 