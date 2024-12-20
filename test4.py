import numpy as np
import matplotlib.pyplot as plt

# Given dataset
dataSet = np.array([
    [0.2331, 2.3385], [1.5207, 2.1946], [0.6499, 1.6730], [0.7757, 1.6365],
    [1.0524, 1.7844], [1.1974, 2.0155], [0.2908, 2.0681], [0.2518, 2.1213],
    [0.6682, 2.4797], [0.5622, 1.5118], [0.9023, 1.9692], [0.1333, 1.8340],
    [-0.5431, 1.8704], [0.9407, 2.2948], [-0.2126, 1.7714], [0.0507, 2.3939],
    [-0.0810, 1.5648], [0.7315, 1.9329], [0.3345, 2.2027], [1.0650, 2.4568],
    [-0.0247, 1.7523], [0.1043, 1.6991], [0.3122, 2.4883], [0.6655, 1.7259],
    [0.5838, 2.0466], [1.1653, 2.0226], [1.2653, 2.3757], [0.8137, 1.7987],
    [-0.3399, 2.0828], [0.5152, 2.0798], [0.7226, 1.9449], [-0.2015, 2.3801],
    [0.4070, 2.2373], [-0.1717, 2.1614], [-1.0573, 1.9235], [-0.2099, 2.2604],
    [1.4010, 1.0298], [1.2301, 0.9611], [2.0814, 0.9154], [1.1655, 1.4901],
    [1.3740, 0.8200], [1.1829, 0.9399], [1.7632, 1.1405], [1.9739, 1.0678],
    [2.4152, 0.8050], [2.5890, 1.2889], [2.8472, 1.4601], [1.9539, 1.4334],
    [1.2500, 0.7091], [1.2864, 1.2942], [1.2614, 1.3744], [2.0071, 0.9387],
    [2.1831, 1.2266], [1.7909, 1.1833], [1.3322, 0.8798], [1.1466, 0.5592],
    [1.7087, 0.5150], [1.5920, 0.9983], [2.9353, 0.9120], [1.4664, 0.7126],
    [2.9313, 1.2833], [1.8349, 1.1029], [1.8340, 1.2680], [2.5096, 0.7140],
    [2.7198, 1.2446], [2.3148, 1.3392], [2.0353, 1.1808], [2.6030, 0.5503],
    [1.2327, 1.4708], [2.1465, 1.1435], [1.5673, 0.7679], [2.9414, 1.1288]
])

def initialize_centroids(data, k):
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices]

def assign_clusters(data, centroids):
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(data, labels, k):
    new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
    return new_centroids

def kmeans(data, k, max_iters=100, tol=1e-4):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iters):
        labels = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, labels, k)
        if np.all(np.abs(new_centroids - centroids) < tol):
            break
        centroids = new_centroids
    return centroids, labels

def plot_clusters(data, centroids, labels, k):
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red')
    plt.title(f'K-Means Clustering with K={k}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Perform K-means clustering for K=2, 3, 5
for k in [2, 3, 5]:
    centroids, labels = kmeans(dataSet, k)
    plot_clusters(dataSet, centroids, labels, k)
