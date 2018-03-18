import csv
import cv2
import os
import re
import numpy
import matplotlib.pyplot as plt
from collections import Counter
from preprocessing import WORKING_PATH

INPUT_PATH = os.path.abspath('./test_data/stage1_train/')
OUTPUT_FILE = os.path.join(WORKING_PATH, 'cellsize_summary.csv')
OUTPUT_PLOT = os.path.join(WORKING_PATH, 'cellsize_summary.png')

# Regex to find target directory for image processing
VALID_PATH_RE = re.compile('.*/stage1_train/[A-Za-z0-9]+$')

# Regex to isolate image label
IMAGE_LABEL_RE = re.compile('.*/stage1_train/([A-Za-z0-9]*).*')

if __name__ == "__main__":

    summary = []

    for root, subFolders, files in os.walk(INPUT_PATH):
        # Find stage1_train/id
        match = VALID_PATH_RE.search(root)
        try:
            match.group(0)
        except AttributeError:
            continue

        image_label = IMAGE_LABEL_RE.search(root).group(1)
        for _root, _, _files in os.walk(os.path.join(root, 'masks')):
            for _file in _files:
                # Get rid of the RGB dimension
                mask_array = cv2.imread(os.path.join(_root, _file))[:,:,0]
                # Locate the cell (with value != 0)
                positive_pos = numpy.where(mask_array)
                (xmin, ymin) = numpy.amin(positive_pos, axis=1)
                (xmax, ymax) = numpy.amax(positive_pos, axis=1)

                # Summary saves the size of the image, the image id, and the mask id
                mask_label = _file.rstrip('.png')
                summary.append([xmax-xmin+1, ymax-ymin+1, image_label, mask_label])

    summary.sort(key=lambda x: x[0]*x[1])
    with open(OUTPUT_FILE, 'w') as f:
        writer = csv.writer(f)
        map(writer.writerow, summary)

    # Extract height, width information
    x, y = zip(*summary)[0:2]

    # Calculate counts for each x*y size and extract x, y and s
    counter = Counter(zip(x, y))
    xy, s = zip(*counter.items())
    x, y = zip(*xy)

    # Use counting information as the size of the dots
    plt.scatter(y, x, s=s)
    plt.xlabel('Length of width')
    plt.ylabel('Length of height')
    plt.title('Distribution of cell sizes of the test dataset')
    # plt.show()
    plt.savefig(os.path.join(OUTPUT_PLOT))
