# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import plotly as py
import plotly.graph_objs as go


# %%
def check(data):
    print(data.shape)  # Row and Columns
    print("\n--------------------------------\n")
    print(data.isnull().sum())  # shows is any null data is in the dataset
    print("\n--------------------------------\n")
    print(data.info())  # Shows both


customer_data = pd.read_csv("Mall_Customers.csv")
check(customer_data)


# %%
def wcss_calculate(wcss_c, data):
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(data)
        wcss_c.append(kmeans.inertia_)  # wcss hesaplar
    return wcss_c


def elbow_graph(wcss_g):
    sns.set()
    plt.plot(range(1, 11), wcss_g)
    plt.title("Elbow Point Graph")
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.show()


X = customer_data.iloc[:, [3, 4]].values  # Data'daki 4 ve 5.'inci sütunların verileri seçildi

wcss = []  # within clusters sum of squares
wcss = wcss_calculate(wcss, X)
elbow_graph(wcss)

# %%
def plot_cluster(X, Y, kmeans_t):
    plt.figure(figsize=(8, 8))
    plt.scatter(X[Y == 0, 0], X[Y == 0, 1], s=50, c="green", label="Cluster 1")
    plt.scatter(X[Y == 1, 0], X[Y == 1, 1], s=50, c="red", label="Cluster 2")
    plt.scatter(X[Y == 2, 0], X[Y == 2, 1], s=50, c="blue", label="Cluster 3")
    plt.scatter(X[Y == 3, 0], X[Y == 3, 1], s=50, c="purple", label="Cluster 4")
    plt.scatter(X[Y == 4, 0], X[Y == 4, 1], s=50, c="gray", label="Cluster 5")

    plt.scatter(kmeans_t.cluster_centers_[:, 0], kmeans_t.cluster_centers_[:, 1], s=100, c="cyan", label="Centroid")
    plt.title("Customer Groups")
    plt.xlabel("Annual Income")
    plt.ylabel("Spending Score")
    plt.show()


km = KMeans(n_clusters=5, init="k-means++", random_state=0)  # K-means modelini yükledik
Y = km.fit_predict(X)
plot_cluster(X, Y, km)

customer_data["Target"] = Y
print(customer_data)

