# import pandas as pd

#HW 1

# df = pd.read_csv('Spring 2025 Courses/ITP 259/EXAM_1_Guide/wineQualityReds.csv')

# print(df.head())

# print(df.sort_values(by = 'volatile.acidity', ascending=False))

# print(df[df['quality'] == 7])

# print(df['pH'].mean())

# print(df[df['alcohol'] > 10].count()) #or
# high_alc = df[df['alcohol'] > 10]
# print(high_alc.shape[0])

# print(df.loc[df['alcohol'].idxmax()])

# random_wine = df.sample(1)
# print(random_wine['residual.sugar'].values[0])

# qual_4 = df[df['quality'] == 4]
# print(qual_4.sample(1))

# df_new = df[df['quality'] != 4]
# print(df_new.shape[0])

#HW 2

# import pandas as pd
# from sklearn.preprocessing import Normalizer
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt

# df = pd.read_csv('Spring 2025 Courses/ITP 259/EXAM_1_Guide/wineQualityReds.csv')

# df.drop('Wine', axis=1, inplace=True)

# quality = df['quality']

# df.drop('quality', axis=1, inplace=True)

# norm = Normalizer()

# df_norm = pd.DataFrame(norm.transform(df), columns = df.columns)

# print(df_norm)

# ks = range(1, 11)
# inertias = []

# for k in ks:
#     model = KMeans(n_clusters=k)
#     model.fit(df_norm)
#     inertias.append(model.inertia_)

# plt.plot(ks, inertias, "-o")
# plt.xlabel("Number of Clusters, k")
# plt.ylabel("Inertia")
# plt.xticks(ks)
# plt.show()

# model = KMeans(n_clusters=6, random_state=2023)
# model.fit(df_norm)
# labels = model.predict(df_norm)
# df_norm['cluster'] = pd.Series(labels)

# df_norm['quality'] = quality
# # print(df_norm)

# print(pd.crosstab(df_norm['quality'], df_norm['cluster']))

#HW 3

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# df = pd.read_csv('Spring 2025 Courses/ITP 259/EXAM_1_Guide/Titanic.csv')

# # print(df.head())
# survived = df['Survived']

# df_new = df.drop(['Passenger', 'Survived'], axis=1)
# df_dummy = pd.get_dummies(df_new, prefix=['Class','Sex', 'Age'])
# # print(df_dummy)

# X = df_dummy 
# y = survived

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2023)

# log_reg = LogisticRegression()
# log_reg.fit(X_train, y_train)

# y_pred = log_reg.predict(X_test)
# # print(metrics.accuracy_score(y_test, y_pred))

# cm = confusion_matrix(y_test, y_pred)

# ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=log_reg.classes_).plot()

# # plt.show()

# # print(df_dummy)
# print(log_reg.predict([[0, 0, 1, 0, 0, 1, 1, 0]]))


from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# diabetes_knn = pd.read_csv('Spring 2025 Courses/ITP 259/EXAM_1_Guide/diabetes.csv')

# print(diabetes_knn.head())
# outcome = diabetes_knn['Outcome']

# X = diabetes_knn.drop('Outcome', axis=1)
# y = outcome

# scaler = StandardScaler()

# scaler.fit(X)
# X_scaled = scaler.transform(X)

# X_train_A, X_temp, y_train_A, y_temp = train_test_split(X_scaled, y, test_size=0.4, random_state=2023, stratify=y)
# X_train_B, X_test, y_train_B, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=2023, stratify=y_temp)

# ks = range(1, 31)
# train_A_scores = []
# train_B_scores = []

# for k in ks:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     knn.fit(X_train_A, y_train_A)
#     y_pred_A = knn.predict(X_train_A)
#     y_pred_B = knn.predict(X_train_B)
#     y_pred = knn.predict(X_test)
#     train_A_scores.append(metrics.accuracy_score(y_train_A, y_pred_A))
#     train_B_scores.append(metrics.accuracy_score(y_train_B, y_pred_B))

# plt.title("KNN: Varying number of neighbors")
# plt.plot(ks, train_A_scores, label="Training A Accuracy")
# plt.plot(ks, train_B_scores, label = "Training B Accuracy")
# plt.legend()
# plt.xlabel("Number of Neighbors")
# plt.ylabel("Accuracy")
# plt.show()

# knn = KNeighborsClassifier(n_neighbors=10)
# knn.fit(X_train_A, y_train_A)
# y_pred = knn.predict(X_test)

# # print(metrics.accuracy_score(y_test, y_pred))

# # cm = confusion_matrix(y_test, y_pred)
# # ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_).plot()

# # plt.show()


# print(knn.predict([[2, 150, 85, 22, 200, 30, 0.3, 55]]))


