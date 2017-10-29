import sys
import random
import math
import matplotlib.pyplot as plt

def parse_args():
    if len(sys.argv) != 3:
        print("Usage: python hyperball.py <dimensions> <points>")
        sys.exit(1)
    dimensions = int(sys.argv[1])
    number_of_points = int(sys.argv[2])
    return dimensions, number_of_points

def prepare_points(dimension, number_of_points):
    points = []
    for _ in range(0, number_of_points):
        point = []
        for _ in range(0, dimension):
            i = random.uniform(0, 1)
            point.append(i)
        points.append(point)
    return points

def distance_to_each_other(points):
    sum = 0
    for point in points:
        dist = 0
        for other in points:
            for i in range(0, len(point)):
                dist += (point[i] - other[i]) ** 2
        sum += math.sqrt(dist)
    return sum / len(points)

def standard_deviation(points, mean_distance):
    sum = 0
    for point in points:
        dist = 0
        for other in points:
            if point != other:
                for i in range(0, len(point)):
                    dist += (point[i] - other[i]) ** 2
        sum += (math.sqrt(dist) - mean_distance) ** 2
    return math.sqrt(sum / len(points))

def main():
    dimensions, number_of_points = parse_args()
    results = []

    for i in range(1, dimensions + 1):
        points = prepare_points(i, number_of_points)
        mean_distance = distance_to_each_other(points)
        deviation = standard_deviation(points, mean_distance)
        res = deviation / mean_distance
        results.append(res)

    ordinals = range(1, len(results) + 1)

    plt.plot(ordinals, results, 'bo', ordinals, results, 'k')
    plt.xticks(ordinals)
    plt.show()

if __name__ == '__main__':
    main()
