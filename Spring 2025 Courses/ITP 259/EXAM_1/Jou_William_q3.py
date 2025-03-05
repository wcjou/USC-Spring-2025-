import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Question 3:

# Reading in dataset
df = pd.read_csv('Spring 2025 Courses/ITP 259/EXAM_1/mushrooms.csv')

# Exploring dataset, target variable is the class variable
# print(df.head())
toxicity = df['class']
df.drop('class', axis=1, inplace=True)
# print(toxicity)

df_dummy = pd.get_dummies(df) 

# no you do not have to encode the target variable

# assigning feature X and target y
X = df_dummy
y = toxicity

# partitioning the X and y and then printing the number of mushrooms in each partition
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, stratify=y, random_state=2023)
# print(X_train.shape[0], X_test.shape[0], y_train.shape[0], y_test.shape[0])


log_reg = LogisticRegression()
# explain point here

log_reg.fit(X_train, y_train)

y_train_pred = log_reg.predict(X_train)
y_pred = log_reg.predict(X_test)

# Printing the training accuracy of the model
# print("Train Accuracy: ", accuracy_score(y_train, y_train_pred))

# Printing the testing accuracy of the model
# print("Test Accuracy: ", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=log_reg.classes_).plot()
# plt.show()

y_probas = log_reg.predict_proba(X_test)
# print(y_probas)
# This output gives the probabilities that each data point would be classfiied
# as either posinous or edible

sample_shroom = {'cap-shape':["x"],
                 'cap-surface':["s"],
                 'cap-color':["n"],
                 'bruises':["t"],
                 'odor':["y"],
                 'gill-attachment':["f"],
                 'gill-spacing':["c"],
                 'gill-size':["n"],
                 'gill-color':["k"],
                 'stalk-shape':["e"],
                 'stalk-root':["e"],
                 'stalk-surface-above-ring':["s"],
                 'stalk-surface-below-ring':["s"],
                 'stalk-color-above-ring':["w"],
                 'stalk-color-below-ring':["w"],
                 'veil-type':["p"],
                 'veil-color':["w"],
                 'ring-number':["o"],
                 'ring-type':["p"],
                 'spore-print-color':["r"],
                 'population':["s"],
                 'habitat':["u"]}

# Plot the decision boundary. Create a mesh of x and y points. Then
# predict the label. Then plot those with color.
X1 = np.arange(-2,2, 0.01) # 400 points
X2 = np.arange(-2,2,0.01) # 400 points

X1, X2 = np.meshgrid(X1, X2)

X_decision = pd.DataFrame({"A": np.reshape(X1,160000), "B": np.reshape(X2,160000)}) # 400*400 = 160000
Z = log_reg.predict(X_decision)

plt.scatter(x=X_decision["A"],y=X_decision["B"], c=Z, cmap="BuGn")
plt.scatter(x=X[:,0], y=X[:,1], c=y)
plt.show()