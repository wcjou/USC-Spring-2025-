# William Jou
# ITP 259 Spring 2025
# HW 3

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import scikitplot as skplt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# 1. Reading dataset into dataframe
titanic_df = pd.read_csv('USC-Spring-2025-/Spring 2025 Courses/ITP 259/Homework/HW3/Titanic.csv')

## 2. Exploring the dataset by printing the head, the target variable is 'Survived'
print(titanic_df.head())

# 3. Dropping irrelevant factors and the target variable
titanic_temp = titanic_df.drop(['Passenger', 'Survived'], axis=1)
print(titanic_temp.head())

# 4. Converting all categorical feature variables into dummy variables
titanic_dummy = pd.get_dummies(titanic_temp, prefix=['Class', 'Sex', 'Age'])
print(titanic_dummy)

# 5. Assigning features to X, and the target variable to y
X = titanic_dummy
y = titanic_df['Survived']

# 6. Splitting the data into testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2023)

# 7. Defining a logistic regression model and fitting it to the training data
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# 8. Using the model to predict with the test features and then measuring the accuracy of the predictions
y_pred = log_reg.predict(X_test)
y_probas = log_reg.predict_proba(X_test)
accuracy_score = metrics.accuracy_score(y_test, y_pred)
print(accuracy_score)

# 9. Plotting the lift curve
skplt.metrics.plot_lift_curve(y_test, y_probas)
plt.show()

# 10. Plotting the confusion matrix 
cnf_matrix = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cnf_matrix, display_labels=log_reg.classes_).plot()
plt.show()

# 11. Using the model to predict the survivability of a male adult passenger traveling in 3rd class
print(log_reg.predict([[0, 0, 1, 0, 0, 1, 1, 0]]))

