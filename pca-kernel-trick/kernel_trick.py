import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from PIL import Image as im

WHITE = 255

def load_image(path):
    img = im.open(path)
    rgb_img = img.convert('RGB')
    points = []
    colours = []

    for x in range(rgb_img.width):
        for y in range(rgb_img.height):
            r, g, b = rgb_img.getpixel((x, y))
            hexc = '#%02x%02x%02x' % (r, g, b)
            if (r, g, b) != (255, 255, 255):
                point = [x, -y + img.height]
                points.append(point)
                colours.append(hexc)

    return points, colours

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)


def main():
    points, colours = load_image('data/set1.bmp')
    X = np.asarray(points)

    pca = PCA()
    X_pca = pca.fit_transform(X)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.2, c=colours, marker='.')

    # # Kernel adjust
    # kpca = KernelPCA(kernel='rbf', gamma=10)
    # X_kpca = kpca.fit_transform(X)
    # plt.scatter(X_kpca[:, 0], X_kpca[:, 1], alpha=0.2, c=colours, marker='.')

    # for length, vector in zip(pca.explained_variance_, pca.components_):
    #     # print(length, vector)
    #     v = vector * 3 * np.sqrt(length)
    #     draw_vector(pca.mean_, pca.mean_ + v)
    # plt.axis('equal');


    plt.show()

if __name__ == '__main__':
    main()
