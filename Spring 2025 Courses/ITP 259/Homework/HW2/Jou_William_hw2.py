# William Jou
# ITP 259 Spring 2025
# HW 2

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import seaborn as sns

winedf = pd.read_csv("Spring 2025 Courses/ITP 259/Homework/HW2/wineQualityReds.csv")

winedf.drop('Wine', axis=1, inplace=True)

wine_quality = winedf['quality']

winedf.drop('quality', axis=1, inplace=True)

# print(winedf)
# print(wine_quality)

norm = Normalizer()

winedf_norm = pd.DataFrame(norm.transform(winedf), columns=winedf.columns)

# print(winedf_norm)


ks = range(1, 11)
inertias = []
for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(winedf_norm)
    inertias.append(model.inertia_)

plt.plot(ks, inertias, "-o")
plt.xlabel("Number of Clusters, k")
plt.ylabel("Inertia")
plt.xticks(ks)
# plt.show()

# You would pick 5 clusters 20%/elbow rule

model = KMeans(n_clusters=5, random_state=2023)
model.fit(winedf_norm)
labels = model.predict(winedf_norm)
winedf_norm['cluster'] = pd.Series(labels)

# print(winedf_norm)

winedf_norm['quality'] = wine_quality

print(pd.crosstab(winedf_norm['quality'], winedf_norm['cluster']))