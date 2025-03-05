import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#1 
df = pd.read_csv('USC-Spring-2025-\Spring 2025 Courses\ITP 259\EXAM_1_Guide\wineQualityReds.csv')
# print(df.head())

#2
df.drop('Wine', axis=1, inplace=True)
print(df)

#3
wine_quality = df['quality']

#4
df.drop('quality', axis=1, inplace=True)

#5
# print(df)
# print(wine_quality)

#6
norm = Normalizer()
df_norm = pd.DataFrame(norm.transform(df), columns=df.columns)

#7
# print(df_norm)

#8
ks = range(1, 11)
inertias = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(df_norm)
    inertias.append(model.inertia_)

#9
plt.plot(ks, inertias, "-o")
plt.xlabel("Number of Clusters, k")
plt.ylabel("Inertia")
plt.xticks(ks)
# plt.show()

#10
#you would choose 5 clusters

#11
model = KMeans(n_clusters=5, random_state=2023)
model.fit(df_norm)
labels = model.predict(df_norm)
df['cluster'] = pd.Series(labels)
# print(df)

#12
df['quality'] = wine_quality
# print(df)

#13
print(pd.crosstab(df['quality'], df['cluster']))