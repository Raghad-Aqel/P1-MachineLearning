import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# read the data set from Excel sheet and clean it
diabetesDataSet = pd.read_csv("C:\\Users\\97059\\PycharmProjects\\Project1-ML\\Diabetes.csv")
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


# Part 2.1
# Apply linear regression to learn the attribute “Age” using all independent attributes (call this model LR1)
print("\nPart 2.1")
X = cleanDataSet.drop('AGE', axis=1)
Y = cleanDataSet['AGE']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=42)
LR1 = LinearRegression()
LR1.fit(X_train, Y_train)
predictions = LR1.predict(X_test)
LR1MSE = mean_squared_error(Y_test, predictions)
LR1RMSE = np.sqrt(LR1MSE)
LR1MAE = mean_absolute_error(Y_test, predictions)
LR1R2 = r2_score(Y_test, predictions)
print('MSE For LR1: ', LR1MSE)
print('RMSE For LR1: ', LR1RMSE)
print('MAE For LR1: ', LR1MAE)
print('R^2 for LR1: ', LR1R2)
plt.scatter(Y_test, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values vs Predictions')
plt.show()



# Part 2.2
# Apply linear regression using the most important feature (based on the correlation matrix) (call this model LR2)
print("\nPart 2.2")
X = cleanDataSet[['NPG']]
Y = cleanDataSet['AGE']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=42)
LR2 = LinearRegression()
LR2.fit(X_train, Y_train)
predictions = LR2.predict(X_test)
LR2MSE = mean_squared_error(Y_test, predictions)
LR2RMSE = np.sqrt(LR2MSE)
LR2MAE = mean_absolute_error(Y_test, predictions)
LR2R2 = r2_score(Y_test, predictions)
print('MSE For LR2: ', LR2MSE)
print('RMSE For LR2: ', LR2RMSE)
print('MAE For LR2: ', LR2MAE)
print('R^2 for LR2: ', LR2R2)
plt.scatter(Y_test, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values vs Predictions')
plt.show()


# Part 2.3
# Apply linear regression using the set of 3-most important features (based on the correlation matrix)
# (call this model LR3)
print("\nPart 2.3")
X = cleanDataSet[['NPG', 'PGL', 'DIA']]
Y = cleanDataSet['AGE']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=42)
LR3 = LinearRegression()
LR3.fit(X_train, Y_train)
predictions = LR3.predict(X_test)
LR3MSE = mean_squared_error(Y_test, predictions)
LR3RMSE = np.sqrt(LR3MSE)
LR3MAE = mean_absolute_error(Y_test, predictions)
LR3R2 = r2_score(Y_test, predictions)
print('MSE For LR3: ', LR3MSE)
print('RMSE For LR3: ', LR3RMSE)
print('MAE For LR3: ', LR3MAE)
print('R^2 for LR3: ', LR3R2)
plt.scatter(Y_test, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values vs Predictions')
plt.show()