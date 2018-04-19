import cv2
import os
import re
import numpy as np
from preprocessing import WORKING_PATH

# ! When specifying absolute path, use /Users/username/ instead of ~/
INPUT_PATH = os.path.abspath('./test_data/stage1_train/')

# Overlay of all masks of the same image
OUTPUT_PATH_MASK = os.path.abspath(os.path.join(WORKING_PATH, 'overlay_mask_only/'))

# Extract original image to one subdirectory
OUTPUT_PATH_IMAGE = os.path.abspath(os.path.join(WORKING_PATH, 'images'))

# Overlay of all masks to the same image, GRB mode
OUTPUT_PATH_IMAGE_MASK_RGB = os.path.abspath(os.path.join(WORKING_PATH, 'overlay_RGB'))

# Overlay of all masks to the same image, Gray mode
OUTPUT_PATH_IMAGE_MASK_GRAY = os.path.abspath(os.path.join(WORKING_PATH, 'overlay_gray'))

# Labels image
OUTPUT_PATH_IMAGE_LABELS = os.path.abspath(os.path.join(WORKING_PATH, 'overlay_labels'))

# Summary of masks numbers of each image
# For visualization purpose
OUTPUT_PATH_SUMMARY_FILE = os.path.abspath(os.path.join(WORKING_PATH, 'cellnumber_summary.csv'))

# Regex to find target directory for image processing
VALID_PATH_RE = re.compile('.*/stage1_train/[A-Za-z0-9]+$')

# Regex to isolate image label
IMAGE_LABEL_RE = re.compile('.*/stage1_train/([A-Za-z0-9]*).*')


def check_path(path):
    """
    Setup the paths for outputs 
    
    :param path: the directory path for output
    """

    if not os.path.isabs(path):
        path = os.path.abspath(path)

    if not os.path.exists(path):
        os.makedirs(path)


def convert_to_labels(labels, img):
    """
    Map from image to nplabels for training
    2: margin
    1: nuclei
    0: background
    
    :param labels: numpy array. has the same shape as img 
    :param img: input matrix
    :return: updated nplabels
    """
    # Narrow down to rectangle with nuclei
    positive_pos = np.where(img)
    (xmin, ymin) = np.amin(positive_pos, axis=1)
    (xmax, ymax) = np.amax(positive_pos, axis=1)

    h, w = img.shape
    for i in range(max(xmin-1, 0), min(xmax+2, h)):
        for j in range(max(ymin-1, 0), min(ymax+2, w)):
            k_xmin,k_xmax = max(i-1,0),min(i+2,h)
            k_ymin,k_ymax = max(j-1,0),min(j+2,w)
            cnt = np.count_nonzero(img[k_xmin:k_xmax, k_ymin:k_ymax])
            # all > 0 : 1. all == 0: 0, else: 2

            if cnt == (k_xmax-k_xmin) * (k_ymax-k_ymin):
                labels[i, j] = 1
            elif cnt > 0:
                labels[i, j] = 2
    return labels


def save_to_csv(content):
    """
    Write to csv files
    
    :param content: the content line to write to the file 
    (mostly list in this case)
    """
    with open(OUTPUT_PATH_SUMMARY_FILE, 'a') as f:
        if isinstance(content, list):
            f.write(','.join(map(str, content)))
        else:
            f.write(str(content))

        f.write('\n')


def image_writer(path, filename, content):
    """
    Export image to file with opencv
    
    :param path: (the list) of output paths
    :param filename: (the list) of target filenames
    :param content: (the list) of the image to output
    """
    if len(path) != len(filename) or len(path) != len(content):
        raise ValueError("check the output paths and contents!")

    for i in range(len(path)):
        cv2.imwrite(os.path.join(path[i], filename[i]), content[i])



def main():
    """
    Recursively looking into data directory
    Process the overlay, write to a different subdirectory,
    while output the statistic information of the dataset  
    """

    total_images = 0

    # Path check
    path_list = [OUTPUT_PATH_IMAGE, OUTPUT_PATH_MASK,
                 OUTPUT_PATH_IMAGE_MASK_RGB, OUTPUT_PATH_IMAGE_MASK_GRAY,
                 OUTPUT_PATH_IMAGE_LABELS]
    map(check_path, path_list)
    if os.path.exists(OUTPUT_PATH_SUMMARY_FILE):
        os.remove(OUTPUT_PATH_SUMMARY_FILE)

    for root, subFolders, files in os.walk(INPUT_PATH):
        # Determine if current path is the useful one
        # (looking for root composed of id numbers)
        # by looking for preset Regex pattern in the path
        match = VALID_PATH_RE.search(root)
        try:
            match.group(0)
        except AttributeError:
            continue

        # Extract image information
        total_images += 1
        image_id = IMAGE_LABEL_RE.search(root).group(1)
        original = cv2.imread(os.path.join(root, 'images', image_id+'.png'))

        # Overlay masks
        masks = None
        # No subdirectory, but files, will be found in path root/masks
        for _root, _, _files in os.walk(os.path.join(root, 'masks')):
            save_to_csv([image_id, len(_files)])
            masks = None
            labels = np.zeros(original.shape[0:2], np.uint8) # No color channel

            for _file in _files:
                foreground = cv2.cvtColor(cv2.imread(os.path.join(_root, _file)), cv2.COLOR_BGR2GRAY)
                if masks is None:
                    masks = foreground
                else:
                    masks = cv2.add(masks, foreground)
                labels = convert_to_labels(labels, foreground)


        # Overlay masks to original images
        overlay_rgb = cv2.add(original, cv2.cvtColor(masks, cv2.COLOR_GRAY2BGR))
        overlay_gray = cv2.add(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), masks)
        # cv2.imshow('%s' %image_id, overlay)

        # Output
        file_list = [image_id+'.png'] * len(path_list)
        content_list = [original, masks, overlay_rgb, overlay_gray, labels*100]
        image_writer(path_list, file_list, content_list)

        # Write nplabels
        nparray_path = os.path.join(INPUT_PATH, image_id, 'nplabels')
        check_path(nparray_path)
        np.save(os.path.join(nparray_path, image_id), labels)

    # print total_images

if __name__ == '__main__':
    main()
