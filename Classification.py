import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# read the data set from Excel sheet and clean it
diabetesDataSet = pd.read_csv("C:\\Users\\97059\\PycharmProjects\\P1-MachineLearning\\Diabetes.csv")
print("\nThe Dataset")
print(diabetesDataSet)

columnsToModify = ['PGL', 'DIA', 'TSF', 'INS', 'BMI', 'DPF', 'AGE']

for column in columnsToModify:
    median = diabetesDataSet[column].median()
    diabetesDataSet[column] = diabetesDataSet[column].replace(0, median)

zScores = np.abs((diabetesDataSet[columnsToModify] - diabetesDataSet[columnsToModify].median()) / diabetesDataSet[columnsToModify].std())
outliers = (zScores > 3)

for column in columnsToModify:
    median = diabetesDataSet[column].median()
    diabetesDataSet.loc[outliers[column], column] = median

cleanDataSet = diabetesDataSet.copy()
print("\n The Clean Dataset")
print(cleanDataSet)

# Part 3.1
# Run k-Nearest Neighbours classifier to predict (the “Diabetic” feature) using the test set
X = cleanDataSet.drop('Diabetic', axis=1)
X = cleanDataSet.iloc[:, 2:11]
X = cleanDataSet.drop(['Diabetic'], axis=1)
Y = cleanDataSet['Diabetic']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

nNeighbors = math.sqrt(len(Y_test))
nNeighbors = int(nNeighbors)
if nNeighbors % 2 == 0:
    nNeighbors = nNeighbors + 1

print('\nN neighbors: ', nNeighbors)
KNN = KNeighborsClassifier(n_neighbors=nNeighbors)
KNN.fit(X_train, Y_train)
predictions = KNN.predict(X_test)
knnAccuracy = accuracy_score(Y_test, predictions)
print('KNN Accuracy: ', knnAccuracy)
confusionMatrix = confusion_matrix(Y_test, predictions)
print('KNN confusion matrix: ')
print(confusionMatrix)

fpr, tpr, thresholds = roc_curve(Y_test, predictions)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('KNN Roc curve')
AUC = roc_auc_score(Y_test, predictions)
print('KNN AUC: ', AUC)
plt.show()


print("\nDifferent Values of K: ")
print("\nk= 9")
nNeighbors = 9
print('N neighbors: ', nNeighbors)
KNN = KNeighborsClassifier(n_neighbors=nNeighbors)
KNN.fit(X_train, Y_train)
predictions = KNN.predict(X_test)
knnAccuracy = accuracy_score(Y_test, predictions)
print('KNN Accuracy: ', knnAccuracy)
confusionMatrix = confusion_matrix(Y_test, predictions)
print('KNN confusion matrix: ')
print(confusionMatrix)

fpr, tpr, thresholds = roc_curve(Y_test, predictions)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('KNN Roc curve')
AUC = roc_auc_score(Y_test, predictions)
print('KNN AUC: ', AUC)
plt.show()


print("\nk= 5")
nNeighbors = 5
print('N neighbors: ', nNeighbors)
KNN = KNeighborsClassifier(n_neighbors=nNeighbors)
KNN.fit(X_train, Y_train)
predictions = KNN.predict(X_test)
knnAccuracy = accuracy_score(Y_test, predictions)
print('KNN Accuracy: ', knnAccuracy)
confusionMatrix = confusion_matrix(Y_test, predictions)
print('KNN confusion matrix: ')
print(confusionMatrix)

fpr, tpr, thresholds = roc_curve(Y_test, predictions)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('KNN Roc curve')
AUC = roc_auc_score(Y_test, predictions)
print('Knn AUC: ', AUC)
plt.show()

print("\nk= 15")
nNeighbors = 15
print('N neighbors: ', nNeighbors)
KNN = KNeighborsClassifier(n_neighbors=nNeighbors)
KNN.fit(X_train, Y_train)
predictions = KNN.predict(X_test)
knnAccuracy = accuracy_score(Y_test, predictions)
print('KNN Accuracy: ', knnAccuracy)
confusionMatrix = confusion_matrix(Y_test, predictions)
print('KNN confusion matrix: ')
print(confusionMatrix)

fpr, tpr, thresholds = roc_curve(Y_test, predictions)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('KNN Roc curve')
AUC = roc_auc_score(Y_test, predictions)
print('KNN AUC: ', AUC)
plt.show()

print("\nk= 7")
nNeighbors = 7
print('N neighbors: ', nNeighbors)
KNN = KNeighborsClassifier(n_neighbors=nNeighbors)
KNN.fit(X_train, Y_train)
predictions = KNN.predict(X_test)
knnAccuracy = accuracy_score(Y_test, predictions)
print('KNN Accuracy: ', knnAccuracy)
confusionMatrix = confusion_matrix(Y_test, predictions)
print('KNN confusion matrix: ')
print(confusionMatrix)

fpr, tpr, thresholds = roc_curve(Y_test, predictions)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('KNN Roc curve')
AUC = roc_auc_score(Y_test, predictions)
print('KNN AUC: ', AUC)
plt.show()