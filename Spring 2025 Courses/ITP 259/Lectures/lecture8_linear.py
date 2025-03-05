# -*- coding: utf-8 -*-
"""Lecture8_Linear.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZxjJe77klAMF9R83QJ7zLXeX62RLbGhR
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier

# make_blobs will create blobs with any number of classes.
# Change the number of centers to change the classes for classification
# similar to make_classification(n_samples=10000, n_features=20, n_informative=5, n_redundant=15, random_state=1) from sklearn.datasets
X, y = make_blobs(n_samples=1000, n_features=2, centers=5, random_state=2023)
# print(X)
# print(X.shape)
# print(y)
# print(y.shape)

# Plot the blobs to visualize the classification problem
plt.scatter(x=X[:,0], y=X[:,1], marker=".", c=y, cmap="RdBu")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Blob distribution')
plt.show()

# Train and test partitions
X_train, X_test , Y_train, Y_test = \
    train_test_split(X, y,test_size=0.30, random_state=2023, stratify=y)
#
# print(X_train)
# print(X_test)
# print(Y_train)
# print(Y_test)

# Instantiate MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(2), activation="relu",
                    max_iter=1000, alpha=1e-3, solver="adam",
                    random_state=2023, learning_rate_init=0.01, verbose=False)

# Training
mlp.fit(X_train, Y_train)

# Plot loss curve
plt.plot(mlp.loss_curve_)
plt.xlabel("Iteration")
plt.ylabel("Crossentropy loss")
plt.show()

# Accuracy of model on test data
# print("The accuracy is", mlp.score(X_test,Y_test))

# Confusion matrix
y_pred = mlp.predict(X_test)

cm = confusion_matrix(Y_test, y_pred, labels=mlp.classes_)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mlp.classes_).plot()
plt.show()

# Plot the decision boundary. Create a mesh of x and y points. Then
# predict the label. Then plot those with color.
X1 = np.arange(-10, 10, 0.1) # 1000 points
X2 = np.arange(-10, 10, 0.1) # 1000 points

X1, X2 = np.meshgrid(X1, X2)

X_decision = pd.DataFrame({"A": np.reshape(X1,1000000), "B": np.reshape(X2,1000000)}) # 1000*1000 = 1000000
Z = mlp.predict(X_decision)

plt.scatter(x=X_decision["A"],y=X_decision["B"], marker = ".", c=Z, cmap="cool")
plt.scatter(x=X[:,0], y=X[:,1], marker = ".", c=y, cmap="RdBu")
plt.show()