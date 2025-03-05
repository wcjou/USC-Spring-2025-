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
print(df.head())
toxicity = df['class']
df.drop('class', axis=1, inplace=True)
print(toxicity)

df_dummy = pd.get_dummies(df) 

# no you do not have to encode the target variable

# assigning feature X and target y
X = df_dummy
y = toxicity

# partitioning the X and y and then printing the number of mushrooms in each partition
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, stratify=y, random_state=2023)
print(X_train.shape[0], X_test.shape[0], y_train.shape[0], y_test.shape[0])

# instantianting model
log_reg = LogisticRegression()
# choosing logistic regression to use sigmoid function so we can get the probailities of each classficiation

# fitting model to training data
log_reg.fit(X_train, y_train)

# predicting with the training and testing data
y_train_pred = log_reg.predict(X_train)
y_pred = log_reg.predict(X_test)

# Printing the training accuracy of the model
print("Train Accuracy: ", accuracy_score(y_train, y_train_pred))

# Printing the testing accuracy of the model
print("Test Accuracy: ", accuracy_score(y_test, y_pred))

# plotting confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=log_reg.classes_).plot()
plt.show()

# printing probas
y_probas = log_reg.predict_proba(X_test)
print(y_probas)
# This output gives the probabilities that each data point would be classfiied
# as either posinous or edible


# predicting toxcicity of sample shroom
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


# converting dict into dataframe
shroom_df = pd.DataFrame(sample_shroom)
# adding sample into original dataframe
df_new = pd.concat([df, shroom_df])
#craeting a new dummy data frame with the added sample
new_dummy = pd.get_dummies(df_new)
#the sample is the last row of the new dataframe, storing it as the sample shroom
sample_shroom = new_dummy.tail(1)

#predicting the toxciity of the sample shroom
print(log_reg.predict(sample_shroom))


# plot decision boundary