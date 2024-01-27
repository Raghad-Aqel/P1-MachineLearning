import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

# read the data set from Excel sheet and clean it
diabetesDataSet =pd.read_csv("C:\\Users\\97059\\PycharmProjects\\P1-MachineLearning\\Diabetes.csv")
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


# Part 1.1
#  print the summary statistics of all attributes in the dataset
print("\nPart 1.1\n")
firstFourColumns = ['NPG', 'PGL', 'DIA', 'TSF']
firstFourColumnsDiabetes = cleanDataSet[firstFourColumns]
summary_statistics1 = firstFourColumnsDiabetes.describe()

secondFiveColumns = ['INS', 'BMI', 'DPF', 'AGE', 'Diabetic']
secondFiveColumnsDiabetes = cleanDataSet[secondFiveColumns]
summary_statistics2 = secondFiveColumnsDiabetes.describe()

print(summary_statistics1)
print("\n")
print(summary_statistics2)


# Part 1.2
# Show the distribution of the class label (Diabetic) and indicate any highlights in the distribution of the class label
print("\nPart 1.2\n")
plt.figure(figsize=(8, 6))
sns.countplot(x='Diabetic', data=cleanDataSet)
print(cleanDataSet[['Diabetic']].value_counts())
plt.xticks([0, 1], ['0 (Not Diabetic)', '1 (Diabetic)'])
plt.xlabel('Diabetic')
plt.ylabel('Count')
plt.title('Distribution of Diabetic Class Label')
plt.show()


# Part 1.3
# For each age group, draw a histogram detailing the amount of diabetics in each sub-group
print("\nPart 1.3\n")
cleanDataSet.hist(column="AGE", by="Diabetic")
plt.xlabel('Diabetic')
plt.ylabel('AGE')
plt.suptitle('Age group and the amount of diabetics')
plt.show()


# Part 1.4
# Show the density plot for the age
print("\nPart 1.4\n")
sns.kdeplot(cleanDataSet['AGE'], fill=True, color='red')
plt.xlabel('AGE')
plt.ylabel('Density')
plt.title('Density Plot for Age')
plt.show()


# Part 1.5
# Show the density plot for the BMI
print("\nPart 1.5\n")
sns.kdeplot(cleanDataSet['BMI'], fill=True, color='red')
plt.xlabel('BMI')
plt.ylabel('Density')
plt.title('Density Plot for BMI')
plt.show()


# Part 1.6
# Visualise the correlation between all features and explain them in your own words
print("\nPart 1.6\n")
correlation_matrix = cleanDataSet.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', fmt=".2f", linewidths=.5)
plt.title('Correlation for all Features')
plt.show()


# Part 1.7
# Split the dataset into training (80%) and test (20%)
print("\nPart 1.7\n")
X = cleanDataSet.drop('Diabetic', axis=1)
y = cleanDataSet['Diabetic']
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training set :", X_train.shape[0])
print("Test set :", X_test.shape[0])
