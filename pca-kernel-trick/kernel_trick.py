import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
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
            hexc = "#%02x%02x%02x" % (r, g, b)
            if (r, g, b) != (255, 255, 255):
                point = [x, -y + img.height]
                points.append(point)
                colours.append(hexc)

    return points, colours



def main():
    points, colours = load_image('data/set2.bmp')

    pca = PCA(n_components=2)
    pca.fit(points)
    pca_points = pca.transform(points)

    x = [x for [x, y] in points]
    y = [y for [x, y] in points]

    # x = pca_points[:,0]
    # y = pca_points[:,1]
    plt.figure(figsize=(7,7))
    plt.scatter(x, y, c=colours, marker=".")
    plt.show()

if __name__ == '__main__':
    main()
