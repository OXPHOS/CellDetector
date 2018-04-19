import cv2
import numpy as np
import os
import pandas as pd
from skimage.morphology import label
from tqdm import tqdm

DATA_DIR = '../test_data/stage1_test'
TMP_DIR = '../test_data/stage1_test_imgs'
OUTPUT_DIR = './prediction_output'
RESIZE_SIZE = 256


class PredictWrapperException(Exception):
    pass


def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def prob_to_rles(x, cutoff=0.0001):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)


class PredictWrapper(object):
    def __init__(self, path, resize_size=256, img_ids=None):
        self.path = path
        self.resize_size = resize_size
        self.img_ids = []
        if img_ids:
            # for test
            self.img_ids = img_ids
        else:
            for name in os.listdir(self.path):
                if os.path.isdir(os.path.join(self.path, name)):
                    self.img_ids.append(name)

        self.img_ids_num = len(self.img_ids)

        # orders are the same with self.img_ids.
        self.resized_images = np.zeros((len(self.img_ids), self.resize_size,
                                        self.resize_size))
        self.original_shapes = []
        print('reading test images...')
        for ind, img_id in tqdm(
                enumerate(self.img_ids), total=self.img_ids_num):

            img = cv2.cvtColor(
                cv2.imread(
                    os.path.join(self.path, img_id, 'images', img_id + '.png')),
                cv2.COLOR_BGR2GRAY)

            if len(img.shape) != 2:
                raise PredictWrapperException(
                    'bad original shape. img_id: %s, shape: %s' % (img_id,
                                                                   img.shape))
            self.original_shapes.append(img.shape)
            self.resized_images[ind] = cv2.resize(img,
                                                  (self.resize_size, self.resize_size))

    def ResizedTestData(self):
        return self.resized_images

    def OutputPrediction(self, prediction, path):
        """prediction shape: (img_num, self.resize_size, self.resize_size)"""
        img_num = len(self.img_ids)
        if prediction.shape != (img_num, self.resize_size, self.resize_size):
            raise PredictWrapperException(
                'prediction is of bad shape. actual shape: %s' %
                (prediction.shape,))
        if not os.path.isdir(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        print('outputing prediction results to ', path)
        for ind, img_id in tqdm(
                enumerate(self.img_ids), total=self.img_ids_num):
            prediction_origin_size = cv2.resize(
                prediction[ind],
                (self.original_shapes[ind][1], self.original_shapes[ind][0]),
                interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(
                os.path.join(path, img_id + '.png'),
                prediction_origin_size)

    def GenerateSubmit(self, prediction, path, cutoff=0.1):
        new_test_ids = []
        rles = []
        for ind, img_id in tqdm(
                enumerate(self.img_ids), total=self.img_ids_num):
            prediction_origin_size = cv2.resize(
                prediction[ind],
                (self.original_shapes[ind][1], self.original_shapes[ind][0]),
                interpolation=cv2.INTER_NEAREST)
            assert prediction_origin_size.shape == self.original_shapes[ind]
            rle = list(prob_to_rles(prediction_origin_size, cutoff=0.1))
            rles.extend(rle)
            new_test_ids.extend([img_id] * len(rle))

        sub = pd.DataFrame()
        sub['ImageId'] = new_test_ids
        sub['EncodedPixels'] = pd.Series(rles).apply(
            lambda x: ' '.join(str(y) for y in x))
        sub.to_csv(os.path.join(path, 'submission.csv'), index=False)

