import numpy as np
import pickle
from sklearn.feature_extraction.text import HashingVectorizer
from  sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

data = pickle.load(open('sklearn-data.pickle', 'rb'))
x_train = data["x_train"]
y_train = data["y_train"]
x_test = data["x_test"]
y_test = data["y_test"]


v#####  Vectorizing data for sklearn
vectorizer = HashingVectorizer(stop_words = "english", lowercase=True,binary=True, n_features = 2**18)
x_train_hash = vectorizer.fit_transform(x_train)
x_test_hash = vectorizer.fit_transform(x_test)


classifier_DT  = DecisionTreeClassifier()
classifier_DT.fit(x_train_hash, y_train)
y_DT = classifier_DT.predict(x_test_hash)


acc_DT = accuracy_score(y_DT, y_test)


print("\nDecision tree accuracy: ", round(acc_DT, 4)*100, "%")