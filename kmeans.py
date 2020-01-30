# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the data set
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Using elbow method to find the optimal number of cluster
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    Kmeans = KMeans(n_clusters = i, init = 'k-means++', n_init = 10, max_iter = 300, random_state = 0)
    Kmeans.fit(X)
    wcss.append(Kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('The number of Cluster')
plt.ylabel('WCSS')
plt.show()
   
# Applying K-Means clustering to the mall dataset
Kmeans = KMeans(n_clusters = 5, init = 'k-means++', n_init = 10, max_iter = 300, random_state = 0)
y_kmeans = Kmeans.fit_predict(X)

# Visualising the results
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0,1], s=100, color = 'red', label = 'careful')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1,1], s=100, color = 'blue', label = 'standart')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2,1], s=100, color = 'yellow', label = 'target')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3,1], s=100, color = 'green', label = 'careless')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4,1], s=100, color = 'magenta', label = 'sensible')
plt.scatter(Kmeans.cluster_centers_[:, 0], Kmeans.cluster_centers_[:, 1], s = 300, label = 'Centroids')
plt.title('Cluster of clients')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()

j = 0
for i in range(1,6):
    plt.scatter(X[y_kmeans == j, 0], X[y_kmeans == j,1], s=100, color = 'yellow', label = 'careful')
    j = j+1
plt.scatter(Kmeans.cluster_centers_[:, 0], Kmeans.cluster_centers_[:, 1], s = 300, label = 'Centroids')
plt.title('Cluster of clients')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()