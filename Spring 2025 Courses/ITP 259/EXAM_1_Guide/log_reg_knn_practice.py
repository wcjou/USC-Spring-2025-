import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

#1
df = pd.read_csv('USC-Spring-2025-\Spring 2025 Courses\ITP 259\EXAM_1_Guide\Titanic.csv')

#2
# print(df.head(10))
survived = df['Survived']
df_temp = df.drop('Survived', axis=1)

#3
df_temp = df_temp.drop('Passenger', axis=1)
# print(df_temp)

#4 
df_dummies = pd.get_dummies(df_temp, prefix=['Class', 'Sex', 'Age'])
# print(df_dummies)

#5
X = df_dummies
y = survived

#6
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2023, stratify=y)

#7
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

#8
y_pred = log_reg.predict(X_test)
# print(accuracy_score(y_test, y_pred))

#9
#skip

#10
# cm = confusion_matrix(y_test, y_pred)
# ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=log_reg.classes_).plot()
# plt.show()

#11
# print(log_reg.predict([[0, 0, 1, 0, 0, 1, 1, 0]]))

#PROBLEM 2

#1
diabetes_knn = pd.read_csv('USC-Spring-2025-\Spring 2025 Courses\ITP 259\EXAM_1_Guide\diabetes.csv')
# print(diabetes_knn.head(10))

#2
outcome = diabetes_knn['Outcome']
diabetes_knn.drop('Outcome', axis=1, inplace=True)

X = diabetes_knn
y = outcome

#3
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

#4
X_train_A, X_temp, y_train_A, y_temp = train_test_split(X_scaled, y, test_size=0.4, random_state=2023, stratify=y)
X_train_B, X_test, y_train_B, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=2023, stratify=y_temp)
# print(X_train_A.shape[0], y_train_A.shape[0], X_train_B.shape[0], y_train_B.shape[0], X_test.shape[0], y_test.shape[0])

#5 & 6
ks = range(1, 31)

training_A_accuracies = []
training_B_accuracies = []

for k in ks:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_A, y_train_A)
    y_pred_A = model.predict(X_train_A)
    y_pred_B = model.predict(X_train_B)
    training_A_accuracy = accuracy_score(y_train_A, y_pred_A)
    training_B_accuracy = accuracy_score(y_train_B, y_pred_B)
    training_A_accuracies.append(training_A_accuracy)
    training_B_accuracies.append(training_B_accuracy)

#7
# plt.title("KNN: Varying number of neighbors")
# plt.plot(ks, training_A_accuracies, label="Training A Accuracy")
# plt.plot(ks, training_B_accuracies, label = "Training B Accuracy")
# plt.legend()
# plt.xlabel("Number of Neighbors")
# plt.ylabel("Accuracy")
# plt.show()

#8
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train_A, y_train_A)
y_pred = knn.predict(X_test)
print(accuracy_score(y_test, y_pred))

#9
# cm = confusion_matrix(y_test, y_pred)
# ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_).plot()
# plt.show()

#10
print(knn.predict([[2, 150, 85, 22, 200, 30, 0.3, 55]]))