import random
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from pyclustering.cluster.kmedians import kmedians
from pyclustering.utils import draw_clusters
from sklearn.decomposition import PCA

k = 3

def init_centers(size, flatten_image):
    # Select random points as clusters' centers
    centers = []
    for center in range(k):
        centers.append(flatten_image[random.randint(0, size - 1)])
    return centers

def init_centers_explicite():
    return [[255, 0, 0], [0, 255, 0], [0, 0, 255]]

def main():
    img = io.imread('monroe.jpg')
    size = img.shape[0] * img.shape[1]
    X = np.asarray(img.reshape(size, 3), dtype='int64')

    centers = init_centers(size, X)
    # centers = init_centers_explicite()

    kmedians_instance = kmedians(X, centers)
    kmedians_instance.process()
    clusters = kmedians_instance.get_clusters()

    X_map = {element: group for group, elts in enumerate(clusters) for element in elts}
    X4 = []
    for i in range(len(X)):
        p = [X[i][0], X[i][1], X[i][2], X_map.get(i)]
        X4.append(p)

    # X4 is the list of [r, g, b, cluster] lists
    # print(X4)
    # draw_clusters(X, clusters)

    pca = PCA(n_components=2)
    pca.fit(X4)
    pca_points = pca.transform(X4)

    # print(pca_points)

    colours = ['#%02x%02x%02x' % (r, g, b) for [r, g, b] in X]

    plt.scatter(pca_points[:, 0], pca_points[:, 1], c=colours, s=10)
    plt.show()

if __name__ == '__main__':
    main()
