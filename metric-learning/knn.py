from PIL import Image as im
from sklearn import neighbors
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import random

WHITE = 255
SPLIT_RATIO = 0.75
k = 3 # Adjust k value

def load_image(path):
    img = im.open(path)
    pixels = []
    for x in range(img.width):
        for y in range(img.height):
            pixel = img.getpixel((x, y))
            if pixel != WHITE:
                p = [x, -y + img.height, pixel]
                pixels.append(p)
    return pixels

def main():
    X = []
    Y = []

    test_X = []
    test_Y = []

    pixels = load_image('data/data256.bmp')
    for record in pixels:
        point = [record[0], record[1]]
        color = record[2]
        if random.random() < SPLIT_RATIO:
            X.append(point)
            Y.append(color)
        else:
            test_X.append(point)
            test_Y.append(color)

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    X_np = np.array(X)

    # clf = neighbors.KNeighborsClassifier(k, weights='distance',
    #       metric='mahalanobis', metric_params={"V": np.cov(X_np.T)})

    clf = neighbors.KNeighborsClassifier(k, 'uniform')
    clf.fit(X, Y)
    h = .1

    x_min, x_max = X_np[:, 0].min() - 1, X_np[:, 0].max() + 1
    y_min, y_max = X_np[:, 1].min() - 1, X_np[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    predictions = clf.predict(test_X)
    accuracy = get_accuracy(test_Y, predictions.tolist())

    plt.figure(figsize=(7,7))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X_np[:, 0], X_np[:, 1], c=Y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

    # Uncomment when measuring accuracy
    # return accuracy

def get_accuracy(test_set_values, predictions):
    length = len(test_set_values)
    correct = 0
    for i in range(length):
        if test_set_values[i] == predictions[i]:
            correct += 1
    return (correct / float(length)) * 100.0

if __name__ == '__main__':
    main()

    # Uncomment when measuring accuracy
    # acc = 0
    # for i in range(10):
    #     acc += main()
    # print(acc / 10)
