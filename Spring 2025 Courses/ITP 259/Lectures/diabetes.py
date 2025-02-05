import pandas as pd
import matplotlib.pyplot as plt

col_names = ["pregnant", "glucose", "bp", "skin", "insulin",
             "bmi", "pedigree", "age", "label"]
print(col_names)
pima = pd.read_csv("Spring 2025 Courses/ITP 259/Lectures/diabetes.csv", header=1, names=col_names)

pd.set_option("display.max_columns", None)
print(pima.head())

feature_cols = ["pregnant", "insulin", "bmi", "age", "glucose", "bp",
                "pedigree", "skin"]
X=pima[feature_cols]
y=pima["label"]
print(X.head())
print(y.head())

# Split X and y into training and testing partitions
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.25, random_state=2022)

from sklearn.linear_model import LogisticRegression

logReg = LogisticRegression(max_iter=1000)

logReg.fit(X_train,y_train)

y_pred = logReg.predict(X_test)

from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

print(cnf_matrix)
print(logReg.classes_)

print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(logReg,X_test,y_test)
plt.show()

