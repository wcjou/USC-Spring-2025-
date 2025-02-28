# William Jou
# ITP 259, Spring 2025
# Homework 4


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Generate spiral data
def generate_spiral(n_points, noise=0.4):
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
X, y = generate_spiral(200)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
plt.title("Spiral Data")
plt.show()