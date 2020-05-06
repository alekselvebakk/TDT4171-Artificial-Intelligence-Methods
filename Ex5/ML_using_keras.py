##Importing libraries for Part2
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Dropout, LSTM





data_keras = pickle.load(open('keras-data.pickle', 'rb'))
x_train_keras = data_keras["x_train"]
y_train_keras = data_keras["y_train"]
x_test_keras = data_keras["x_test"]
y_test_keras = data_keras["y_test"]
vocab_size = data_keras["vocab_size"]
max_length = data_keras["max_length"]



x_train_fixed_length = pad_sequences(x_train_keras, maxlen=max_length, padding='post')
x_test_fixed_length = pad_sequences(x_test_keras, maxlen=max_length, padding='post')

output_size = 100
model = Sequential()
model.add(Embedding(vocab_size, output_size, mask_zero = True, input_length = max_length))
model.add(LSTM(output_size, return_sequences=False))
model.add(Dense(1, activation = 'sigmoid'))
model.compile('adam', loss='mean_squared_error', metrics=['accuracy'])

num_epochs = 3
model.fit(x_train_fixed_length, y_train_keras, verbose=1, epochs = num_epochs, batch_size = 2**9)

result = model.evaluate(x_test_fixed_length, y_test_keras, verbose = 1)
print(result)
