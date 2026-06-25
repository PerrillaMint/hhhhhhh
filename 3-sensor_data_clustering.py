# Demo 3: Clustering Sensor Data
# Goal: Group sensor readings without using labels

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------
# 1. Create a small synthetic sensor dataset
# ----------------------------------------------------
data = {
    "temperature": [35, 36, 34, 37, 38, 39, 75, 80, 78, 85, 90, 88],
    "vibration":   [0.20, 0.25, 0.22, 0.30, 0.28, 0.35, 1.50, 1.70, 1.60, 1.90, 2.10, 2.00],
    "current":     [5.1, 5.0, 5.2, 5.3, 5.1, 5.4, 8.5, 8.8, 8.6, 9.0, 9.3, 9.1]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

# ----------------------------------------------------
# 2. Select input features
# ----------------------------------------------------
X = df[["temperature", "vibration", "current"]]

# ----------------------------------------------------
# 3. Scale the data
# ----------------------------------------------------
# Scaling is important because temperature, vibration, and current
# have different numerical ranges.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------------------------
# 4. Apply K-Means clustering
# ----------------------------------------------------
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to the dataframe
df["cluster"] = clusters

print("\nDataset with cluster labels:")
print(df)

# ----------------------------------------------------
# 5. Visualize the clusters
# ----------------------------------------------------
plt.figure(figsize=(8, 5))

plt.scatter(
    df["temperature"],
    df["vibration"],
    c=df["cluster"],
    s=80
)

plt.xlabel("Temperature")
plt.ylabel("Vibration")
plt.title("Clustering Sensor Data using K-Means")
plt.grid(True)
plt.show()

# ----------------------------------------------------
# 6. Interpret cluster centers
# ----------------------------------------------------
# Cluster centers are in scaled form, so we convert them back
# to the original units.
centers_original_units = scaler.inverse_transform(kmeans.cluster_centers_)

centers_df = pd.DataFrame(
    centers_original_units,
    columns=["temperature", "vibration", "current"]
)

print("\nCluster centers in original units:")
print(centers_df)