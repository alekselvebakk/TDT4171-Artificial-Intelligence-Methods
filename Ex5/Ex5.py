
import pickle

##Importing libraries for Part1
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier

##Importing libraries for Part2
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Dropout, LSTM

################ PART 1 ################

#####  Extracting data
data = pickle.load(open('sklearn-data.pickle', 'rb'))
x_train = data["x_train"]
y_train = data["y_train"]
x_test = data["x_test"]
y_test = data["y_test"]

#####  Vectorizing data for sklearn
vectorizer = HashingVectorizer(stop_words = "english", lowercase=True,binary=True, n_features = 2**18)
x_train_hash = vectorizer.fit_transform(x_train)
x_test_hash = vectorizer.fit_transform(x_test)



#####  Classifier using Naive bayes
classifier_NB = BernoulliNB()
classifier_NB.fit(x_train_hash, y_train)
y_NB = classifier_NB.predict(x_test_hash)


#####  Evaluating NB-model
acc_NB = accuracy_score(y_NB, y_test)
print("\n\nNaive bayes accuracy: \t", round(acc_NB, 4)*100, "%")

#####  Classifier using decision tree
classifier_DT  = DecisionTreeClassifier()
classifier_DT.fit(x_train_hash, y_train)
y_DT = classifier_DT.predict(x_test_hash)

#####  Evaluating DT-model
acc_DT = accuracy_score(y_DT, y_test)

print("\nDecision tree accuracy: ", round(acc_DT, 4)*100, "%")





################ PART 2 ################

#####  Extracting data
data_keras = pickle.load(open('keras-data.pickle', 'rb'))
x_train_keras = data_keras["x_train"]
y_train_keras = data_keras["y_train"]
x_test_keras = data_keras["x_test"]
y_test_keras = data_keras["y_test"]
vocab_size = data_keras["vocab_size"]
max_length = data_keras["max_length"]


#####  Making all data have the same length
x_train_fixed_length = pad_sequences(x_train_keras, maxlen=max_length, padding='post')
x_test_fixed_length = pad_sequences(x_test_keras, maxlen=max_length, padding='post')


#####  Creating the model
output_size = 100
model = Sequential()
model.add(Embedding(vocab_size, output_size, mask_zero = True, input_length = max_length))
model.add(LSTM(output_size, return_sequences=False))
model.add(Dense(1, activation = 'sigmoid'))
model.compile('adam', loss='mean_squared_error', metrics=['accuracy'])

#####  Training the model
num_epochs = 3
model.fit(x_train_fixed_length, y_train_keras, verbose=1, epochs = num_epochs, batch_size = 2**9)


#####  Evaluating model
result = model.evaluate(x_test_fixed_length, y_test_keras, verbose = 1)
print(result)

