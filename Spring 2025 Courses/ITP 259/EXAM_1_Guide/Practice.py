import pandas as pd

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

import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv('Spring 2025 Courses/ITP 259/EXAM_1_Guide/wineQualityReds.csv')

df.drop('Wine', axis=1, inplace=True)

quality = df['quality']

df.drop('quality', axis=1, inplace=True)

norm = Normalizer()

df_norm = pd.DataFrame(norm.transform(df), columns = df.columns)

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

