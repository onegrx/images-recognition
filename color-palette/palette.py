import random
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from pyclustering.cluster.kmedians import kmedians

k = 4

def init_centers(size, flatten_image):
    # Select random points as clusters' centers
    centers = []
    for center in range(k):
        centers.append(flatten_image[random.randint(0, size - 1)])
    return centers

def main():
    img = io.imread('kiwi256.jpg')
    size = img.shape[0] * img.shape[1]
    flatten = np.asarray(img.reshape(size, 3), dtype='int64')

    centers = init_centers(size, flatten)
    print(centers)

    kmedians_instance = kmedians(flatten, centers)
    kmedians_instance.process()
    clusters = kmedians_instance.get_clusters()

if __name__ == '__main__':
    main()
