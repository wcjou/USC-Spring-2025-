# Question 2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

# reading data set in and printing head
df = pd.read_csv('Spring 2025 Courses/ITP 259/EXAM_1/wineQualityReds.csv')
print(df.head())

# dropping irrelevant factors
df_temp = df.drop('Wine', axis=1)
print(df_temp.head())

#storing target variable
quality = df['quality']

#dropping target variable from feature matrix
df_temp = df_temp.drop('quality', axis=1)
print(df_temp.head())
# assigning variables for feature matrix and target vector
X = df_temp
y = quality

print(y)

# scaling data
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
print(X_scaled)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=2023)

#instantiating and fitting model

mlp = MLPClassifier(hidden_layer_sizes=(8, 8, 8, 8, 8, 8), activation='relu', solver='adam', max_iter=1000, random_state=2023)
mlp.fit(X_train, y_train)

# Plotting the loss curve
plt.plot(mlp.loss_curve_)
plt.xlabel("Iteration")
plt.ylabel("Crossentropy loss")
plt.title("Loss Curve")
plt.show()

# Printing the training accuracy of the model
y_train_pred = mlp.predict(X_train)
print("Train Accuracy: ", accuracy_score(y_train, y_train_pred))

# Printing the testing accuracy of the model
y_pred = mlp.predict(X_test)
print("Test Accuracy: ", accuracy_score(y_test, y_pred))

# The training accuracy is slightly more accurate, but this is expected as the model is fitting
# itself to the training data

#plotting confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mlp.classes_).plot()
plt.show()

# predicting the quality of a given wine sample
wine_sample = [[8, 0.6, 0, 2.0, 0.076, 10, 30, 0.9978, 3.20, 0.5, 10.0]]
scaler = StandardScaler()
scaler.fit(wine_sample)
wine_scaled = scaler.transform(wine_sample)
print(mlp.predict(wine_scaled))


