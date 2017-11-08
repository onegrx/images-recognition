import sys
import random
import math
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

RED = '#FF0000'
GREEN = '#00FF00'
BLUE = '#0000FF'
radius = 1

def parse_args():
    if len(sys.argv) != 3:
        print("Usage: python pca_hyperball.py <dimension> <points>")
        sys.exit(1)
    dimension = int(sys.argv[1])
    number_of_points = int(sys.argv[2])
    return dimension, number_of_points

def prepare_points(dimension, radius, number):
    points = []
    for _ in range(0, number):
        point = []
        for _ in range(0, dimension):
            i = random.uniform(-radius, radius)
            point.append(i)
        points.append(point)
    return points

def distance_from_centre(point):
    sum = 0
    for i in point:
        sum += i ** 2
    return math.sqrt(sum)

def main():
    dimension, number_of_points = parse_args()
    points = prepare_points(dimension, radius, number_of_points)
    colours = []

    for point in points:
        if distance_from_centre(point) > radius:
            colours.append(BLUE)
        else:
            colours.append(GREEN)

    corners = [list(i) for i in itertools.product([-radius, radius], repeat=dimension)]

    for corner in corners:
        points.append(corner)
        colours.append(RED)

    pca = PCA(n_components=2)
    pca.fit(points)
    pca_points = pca.transform(points)

    # print("Points: ({})\n".format(len(points)), points)
    # print("Points after PCA: ({})\n".format(len(pca_points)), pca_points)
    print("Points: {}\nCorners: {}".format(len(points), len(corners)))

    x = pca_points[:,0]
    y = pca_points[:,1]
    plt.scatter(x, y, c=colours)
    plt.show()


if __name__ == '__main__':
    main()
