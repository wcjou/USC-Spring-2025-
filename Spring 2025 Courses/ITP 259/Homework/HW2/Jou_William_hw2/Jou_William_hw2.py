# William Jou
# ITP 259 Spring 2025
# HW 2

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import seaborn as sns


# Reading in csv file as dataframe and printing the header
winedf = pd.read_csv("USC-Spring-2025-/Spring 2025 Courses/ITP 259/Homework/HW2/Jou_William_hw2/wineQualityReds.csv")
print(winedf.head())

# Dropping the wine column and reprinting the header
winedf.drop('Wine', axis=1, inplace=True)
print(winedf.head())

# Storing the wine quality column as its own variable and printing it
wine_quality = winedf['quality']
print(wine_quality)


# Dropping the quality column from the dataframe
winedf.drop('quality', axis=1, inplace=True)

# Printing the wine dataframe and the wine quality
print(winedf)
print(wine_quality)

# Creating a normalizer object
norm = Normalizer()

# Normalizing the wine dataframe and storing it as a new dataframe
winedf_norm = pd.DataFrame(norm.transform(winedf), columns=winedf.columns)

# Printing the normalized dataframe
print(winedf_norm)

# Creating a list of k values and a list of inertias and looping through them fit the model
ks = range(1, 11)
inertias = []
for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(winedf_norm)
    inertias.append(model.inertia_)

# Plotting the intertias vs number of clusters graph
plt.plot(ks, inertias, "-o")
plt.xlabel("Number of Clusters, k")
plt.ylabel("Inertia")
plt.xticks(ks)
plt.show()

# You would pick 5 clusters 20%/elbow rule

# Fitting the model with 5 clusters and adding the cluster labels as a column to the normalized dataframe
model = KMeans(n_clusters=5, random_state=2023)
model.fit(winedf_norm)
labels = model.predict(winedf_norm)
winedf_norm['cluster'] = pd.Series(labels)

# Printing the new normalized data frame
print(winedf_norm)

# Adding back the quality column to the normalized dataframe
winedf_norm['quality'] = wine_quality

# Printing a crosstab of quality and cluster
print(pd.crosstab(winedf_norm['quality'], winedf_norm['cluster']))

# The clusters don't seem to be strongly related to the quality of the wine