import csv
import cv2
import os
import matplotlib.pyplot as plt
from collections import Counter
from preprocessing import WORKING_PATH

INPUT_PATH = os.path.join(WORKING_PATH, 'images')
OUTPUT_FILE = os.path.join(WORKING_PATH, 'imagesize_summary.csv')
OUTPUT_PLOT = os.path.join(WORKING_PATH, 'imagesize_summary.png')

if __name__ == "__main__":

    # List/Counter data structure to store image sizes
    image_size_summary = []

    for root, subFolders, files in os.walk(INPUT_PATH):
        for file in files:
            # Extract the first two dimensions (H * W) of each image
            try:
                image_size = tuple(cv2.imread(os.path.join(INPUT_PATH, file)).shape[0:2])
                image_size_summary.append(image_size)
            except AttributeError:
                continue

    image_size_summary = Counter(image_size_summary)

    # Save the sum of images of different sizes to file
    with open(OUTPUT_FILE, 'w') as f:
        writer = csv.writer(f)
        map(writer.writerow, image_size_summary.items())

    # Extract height, width and counts information
    # and scatter plot
    xy, s = zip(*image_size_summary.items())
    x, y = zip(*xy)
    plt.scatter(y, x, s=s)

    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('Length of width')
    plt.ylabel('Length of height')
    plt.title('Distribution of image sizes of the test dataset')

    # plt.show()
    plt.savefig(os.path.join(OUTPUT_PLOT))
