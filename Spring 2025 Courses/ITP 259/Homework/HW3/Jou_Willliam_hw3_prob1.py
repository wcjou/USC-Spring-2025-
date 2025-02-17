# William Jou
# ITP 259 Spring 2025
# HW 3

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

titanic_df = pd.read_csv('USC-Spring-2025-/Spring 2025 Courses/ITP 259/Homework/HW3/Titanic.csv')

print(titanic_df.head())

titanic_temp = titanic_df.drop('Passenger', axis=1)

print(titanic_temp.head())