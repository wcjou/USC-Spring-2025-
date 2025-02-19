# William Jou
# ITP 259 Spring 2025
# HW 3

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. Reading in the data set into a dataframe
diabetes_knn = pd.read_csv('Spring 2025 Courses/ITP 259/Homework/HW3/diabetes.csv')
print(diabetes_knn.head())

# 2. Creating the feature matrix and target vector
X = diabetes_knn.drop('Outcome', axis=1)
y = diabetes_knn['Outcome']
print(X.head())
print(y.head())

# 3. Standardizing the features matrix
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
print(X_scaled)

# 4. Splitting the feature matrix and target vector into 3 partitions
X_train_A, X_temp, y_train_A, y_temp = train_test_split(X_scaled, y, test_size=0.4, random_state=2023, stratify=y)
X_train_B, X_test, y_train_B, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=2023, stratify=y_temp)
print(X_train_A.shape[0], y_train_A.shape[0], X_train_B.shape[0], y_train_B.shape[0], X_test.shape[0], y_test.shape[0])


# 5. & 6. Developing knn model on training A for ks 1-30 and computing the accuracy of the predictions
kns = range(1, 31)
train_A_scores = []
train_B_scores = []

for k in kns:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_A, y_train_A)
    y_pred_A = knn.predict(X_train_A)
    y_pred_B = knn.predict(X_train_B)
    y_pred = knn.predict(X_test)
    train_A_scores.append(metrics.accuracy_score(y_train_A, y_pred_A))
    train_B_scores.append(metrics.accuracy_score(y_train_B, y_pred_B))

print(train_A_scores)
print(train_B_scores)


# # 7. Plotting a graph of the training accuracy for training A and training B over different ks (would choose ~10 ks)
plt.title("KNN: Varying number of neighbors")
plt.plot(kns, train_A_scores, label="Training A Accuracy")
plt.plot(kns, train_B_scores, label = "Training B Accuracy")
plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.show()

# # 8. Creating a new knn model that uses 10 ks and then scoring the model with the test data
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_test, y_test)
y_pred_test = knn.predict(X_test)
test_accuracy = metrics.accuracy_score(y_test, y_pred_test)
print(test_accuracy)

# # 9. Plotting a confusion matrix of the model
cnf_matrix_test = confusion_matrix(y_test, y_pred_test)
ConfusionMatrixDisplay(confusion_matrix=cnf_matrix_test, display_labels=knn.classes_).plot()
plt.show()

# # 10. Using the model to predict the outcome of diabetes for a person with 2 pregnancies, 150 glucose, 85 blood pressure, 22 skin thickness, 200 insulin, 30 BMI, 0.3 diabetes pedigree, 55 age
print(knn.predict([[2, 150, 85, 22, 200, 30, 0.3, 55]]))
