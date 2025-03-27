import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score


# Load the data
df = pd.read_csv('USC-Spring-2025-/Spring 2025 Courses/DSO 445/Lecture/kmeans/fake_bills.csv', delimiter=';')
# print(df.head())
# drop the first column
df = df.drop(df.columns[0], axis=1)
# drop all rows with missing values
df = df.dropna()
# print(df.head())


scaler = StandardScaler()
df_norm = scaler.fit_transform(df)
df_norm = pd.DataFrame(df_norm, columns=df.columns)

print(df_norm.head())

# Define a range of clusters
cluster_range = range(1, 21)

# Compute KMeans for each number of clusters
inertias = []
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42).fit(df_norm)
    inertias.append(kmeans.inertia_)

# Plot the Elbow method
plt.figure(figsize=(10,6))
plt.plot(cluster_range, inertias, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

db_scores = []
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42).fit(df_norm)
    if k > 1:  # davies_bouldin_score requires at least 2 clusters
        score = davies_bouldin_score(df_norm, kmeans.labels_)
        db_scores.append(score)
    else:
        db_scores.append(None)

# Plot the Davies-Bouldin scores
plt.figure(figsize=(10,6))
plt.plot(cluster_range, db_scores, marker='o', linestyle='--')
plt.title('Davies-Bouldin Scores vs. Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Davies-Bouldin Score')
plt.grid(True)
plt.show()



