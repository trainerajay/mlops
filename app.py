from sklearn import datasets
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from utils import split_data


df = pd.read_csv("./data/Iris.csv")
X_train, X_test, y_train, y_test = split_data(df)
#step-1: initialise the model class
clf = DecisionTreeClassifier(criterion="entropy") #Information gain as criteria
#step-2: train the model on training set
clf.fit(X_train,y_train)
#step-3 evaluate the data on testing set
y_pred = clf.predict(X_test)

print(f"Accuacy of the model is {accuracy_score(y_test,y_pred)*100}") #--> this test accuracy


