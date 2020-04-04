import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

file_data = 'data/diabetes.csv'

data_df = pd.read_csv(file_data)
data_df.dropna()
X = data_df.drop(['Outcome'], axis=1).to_numpy()
Y = data_df['Outcome'].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
print("Training size: ", len(y_train))
print("Test size    : ", len(y_test))

knn_clf = KNeighborsClassifier(n_neighbors=10)
knn_clf.fit(x_train, y_train)
y_pred = knn_clf.predict(x_test)
print('KNN(n_neighbors=10)\nAccuracy: ', accuracy_score(y_test, y_pred))


knn_clf = KNeighborsClassifier(n_neighbors=10, p=2)
knn_clf.fit(x_train, y_train)
y_pred = knn_clf.predict(x_test)
print('KNN(n_neighbors=10, p=2)\nAccuracy: ', accuracy_score(y_test, y_pred))

knn_clf = KNeighborsClassifier(n_neighbors=10, p=2, weights='distance')
knn_clf.fit(x_train, y_train)
y_pred = knn_clf.predict(x_test)
print('KNN(n_neighbors=10, p=2, weights="distance"\nAccuracy: ', accuracy_score(y_test, y_pred))

