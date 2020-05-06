import numpy as np
import pickle
from sklearn.feature_extraction.text import HashingVectorizer
from  sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB

data = pickle.load(open('sklearn-data.pickle', 'rb'))
x_train = data["x_train"]
y_train = data["y_train"]
x_test = data["x_test"]
y_test = data["y_test"]

#####  Vectorizing data for sklearn
vectorizer = HashingVectorizer(stop_words = "english", lowercase=True,binary=True, n_features = 2**18)
x_train_hash = vectorizer.fit_transform(x_train)
x_test_hash = vectorizer.fit_transform(x_test)

classifier_NB = BernoulliNB()


classifier_NB.fit(x_train_hash, y_train)


y_NB = classifier_NB.predict(x_test_hash)



acc_NB = accuracy_score(y_NB, y_test)

print("\nNaive bayes accuracy: \t", round(acc_NB, 4)*100, "%")