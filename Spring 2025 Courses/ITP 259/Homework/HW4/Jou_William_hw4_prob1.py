# William Jou
# ITP 259, Spring 2025
# Homework 4

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Function for generating spiral data
def generate_spiral(n_points, noise=1):
    theta = np.linspace(0, 2 * np.pi, n_points)
    r1 = 2 * theta + np.pi
    r2 = -2 * theta - np.pi
    
    x1 = r1 * np.cos(theta) + np.random.normal(0, noise, n_points)
    y1 = r1 * np.sin(theta) + np.random.normal(0, noise, n_points)
    
    x2 = r2 * np.cos(theta) + np.random.normal(0, noise, n_points)
    y2 = r2 * np.sin(theta) + np.random.normal(0, noise, n_points)
    
    X = np.vstack((np.column_stack((x1, y1)), np.column_stack((x2, y2))))
    y = np.hstack((np.zeros(n_points), np.ones(n_points)))
    
    return X, y

# Generate and plot data
X, y = generate_spiral(400)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
plt.show()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=2023)

# Defining and fitting the model
mlp = MLPClassifier(hidden_layer_sizes=(8, 8, 8, 8, 8, 8), activation='relu', solver='adam', max_iter=1000, random_state=2023)
mlp.fit(X_train, y_train)

# Plotting the loss curve
plt.plot(mlp.loss_curve_)
plt.xlabel("Iteration")
plt.ylabel("Crossentropy loss")
plt.title("Loss Curve")
plt.show()

# Printing the Accuracy of the model
y_pred = mlp.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))

# Plotting the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mlp.classes_).plot()
plt.show()

# Plotting the Decision Boundary
xx = np.linspace(-20, 20, 400)
yy = np.linspace(-20, 20, 400)
gx, gy = np.meshgrid(xx, yy)
Z = mlp.predict(np.c_[gx.ravel(), gy.ravel()])
Z = Z.reshape(gx.shape)
plt.contourf(gx, gy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

axes = plt.gca()
axes.set_xlim([-20, 20])
axes.set_ylim([-20, 20])
plt.grid('off')
plt.axis('off')

plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.coolwarm)
plt.title('Model predictions on our Test set')
plt.show()