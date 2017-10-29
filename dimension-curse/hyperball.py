import sys
import random
import math
import matplotlib.pyplot as plt

def parse_args():
    if len(sys.argv) != 4:
        print("Usage: python hyperball.py <dimensions> <radius> <points>")
        sys.exit(1)
    dimensions = int(sys.argv[1])
    radius = float(sys.argv[2])
    number_of_points = int(sys.argv[3])
    return dimensions, radius, number_of_points

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
    dimensions, radius, number_of_points = parse_args()
    results = []

    for i in range(1, dimensions + 1):
        inside = 0
        outside = 0
        points = prepare_points(i, radius, number_of_points)

        for point in points:
            if distance_from_centre(point) > radius:
                outside += 1
            else:
                inside += 1

        percent_inside = inside / number_of_points * 100
        results.append(percent_inside)

    ordinals = range(1, len(results) + 1)
    plt.plot(ordinals, results, 'bo', ordinals, results, 'k')
    plt.xticks(ordinals)
    plt.show()


if __name__ == '__main__':
    main()
