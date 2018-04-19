"""
Given training data path and window size
- Read all the training samples
- Perform augmentation on training samples
- Reshape augmented images to a batch of input
- Return the batch derived from 1 image upon query

To use:
- call imageparser.process_image(inputpath, outputpath, chunksize)
"""

import numpy as np
import os
import cv2
import imgaug as ia
from imgaug import augmenters as iaa
import checkpath
import BatchReader
import imagewriter


INPUT_DIR = os.path.abspath('../test_data/stage1_train/')
TEST_OUTPUT_DIR = os.path.abspath('./test_aug/')
OUTPUT_DIR = os.path.abspath('../test_data/stage1_train_augmentation')

def augmentation(imgs, test=False):
    """
    Perform:
        - Color inversion
        - GaussianBlur
        - Sharpen
        - GaussianNoise
    on input images via imgaug
    
    :param imgs: ndarray with shape [batch_size, rows, cols, channels] 
    :param test: Whether the method is called for test purpose
    :return: augmented images with shape [batch_size, rows, cols]
    """
    # Invert color and add the new image to input
    # Deprecated because of bad performance
    # input_imgs = np.concatenate([imgs, iaa.Invert(1).augment_images(imgs)], axis=0)
    # input_imgs = imgs

    # Augmentation
    aug = [
        iaa.GaussianBlur(sigma=(0.5, 1.0)),  # blur images with a sigma of 0.5 to 1.0
        iaa.Sharpen(alpha=(0.3, 0.5), lightness=1),
        # Sharpen an image, then overlay the results with the original using an alpha between 0.3 and 0.5
        iaa.AdditiveGaussianNoise(scale=(0.01 * 255, 0.03 * 255)),
        # Add noise to an image, sampled once per pixel from a normal distribution N(0.01*255, 0.03*255)
        ]

    aug_imgs = np.concatenate([imgs,
                               np.concatenate(list(map(lambda f:f.augment_images(imgs), aug)),
                                              axis=0)],
                              axis=0)
    aug_imgs = np.squeeze(aug_imgs)

    if test:
        # aug_imgs.shape=[tile, row, col]
        imagewriter.image_writer(TEST_OUTPUT_DIR, aug_imgs, 'aug')

    return aug_imgs


def split_and_concatenate_arrays(array, chunk_size):
    """
    Split full input image to window_size * window_size
    
    :param array: Input image
    :return: image piles
    """
    nrows, ncols = array.shape[1:3]
    array = np.concatenate(np.split(array, nrows//chunk_size, axis=1), axis=0)
    array = np.concatenate(np.split(array, ncols//chunk_size, axis=2), axis=0)
    return array


def next_image(path, image_id, chunk_size, test=False):
    """
    Stream in next image in queue
    Augment and split images, tile and split labels
    
    :param path: path to the image
    :param image_id: id of the image
    :param chunk_size: the size of the image chunk to be output
    :param test: Whether the method is called for test purpose
    :return: split images, split labels ready for training
    """

    # Read image
    raw_img = cv2.cvtColor(
        cv2.imread(os.path.join(path, image_id, 'images', image_id + '.png')),
        cv2.COLOR_BGR2GRAY)
    raw_label = np.load(os.path.join(path, image_id, 'nplabels', image_id + '.npy'))

    # Wrap image
    nrows, ncols = raw_img.shape
    nrows_wrap = chunk_size - nrows % chunk_size if nrows % chunk_size else 0
    ncols_wrap = chunk_size - ncols % chunk_size if ncols % chunk_size else 0
    wrap_img = cv2.copyMakeBorder(raw_img, 0, nrows_wrap, 0, ncols_wrap,
                                  cv2.BORDER_REFLECT)
    wrap_label = cv2.copyMakeBorder(raw_label, 0, nrows_wrap, 0, ncols_wrap,
                                    cv2.BORDER_REFLECT)
    nrows, ncols = wrap_img.shape

    # Image augmentation
    aug_img = augmentation(wrap_img.reshape(1, nrows, ncols, 1), test)

    if test:
        print("----- New Image : %s -----" %image_id)
        print('aug_img.shape: ', aug_img.shape)

    wrap_label = np.expand_dims(wrap_label, 0)
    if test:
        print('wrap_label_img.shape: ', wrap_label.shape)

    # Partition image to chunk_size * chunk_size
    split_img = split_and_concatenate_arrays(aug_img, chunk_size)
    split_label = split_and_concatenate_arrays(wrap_label, chunk_size)

    if test:
        print('split_img.shape: ', split_img.shape)
        print('split_label.shape: ', split_label.shape)
        imagewriter.image_writer(TEST_OUTPUT_DIR, split_img, prefix='split_input')
        imagewriter.image_writer(TEST_OUTPUT_DIR, split_label*100, prefix='split_label')

    return split_img, split_label

def process_image(input_path, output_path, chunk_size, test=False):
    """
    :param pathin: Path to training samples 
    :param pathin: Path to output numpy arrays
    :param chunk_size: the dimension of target outputs
    """
    images = [f for f in os.listdir(input_path) if not f.startswith('.')]

    checkpath.check_path(output_path, ['images', 'labels'])
    if test:
        checkpath.check_path(TEST_OUTPUT_DIR)

    for (cnt, image_id) in enumerate(images):
        if test and cnt == 5:
            break

        img, label = next_image(input_path, image_id, chunk_size, test)
        total_imgs = img.shape[0]
        total_labels = label.shape[0]
        num_aug = total_imgs // total_labels
        for i in range(0, total_labels):
            for j in range(num_aug):
                np.save(os.path.join(
                    output_path, 'images', ''.join([image_id, '_', str(i), '.', str(j)])),
                    img[i*num_aug+j,:,:])
            np.save(os.path.join(
                output_path, 'labels', ''.join([image_id, '_', str(i)])),
                label[i,:,:])

        if test:
            print("\n\nTEST RELOAD")
            batch_reader = BatchReader.BatchReader(output_path, test=test)
            img, label = batch_reader.next_batch(3)
            imagewriter.image_writer(TEST_OUTPUT_DIR, img, prefix='reload_input_iamge')
            imagewriter.image_writer(TEST_OUTPUT_DIR, label*100, prefix='reload_input_label')
            print("\n\n")


if __name__ == "__main__":
    ia.seed(3)
    process_image(INPUT_DIR, OUTPUT_DIR, 128, test=True)