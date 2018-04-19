import numpy as np
import os
from random import shuffle

INPUT_DIR = os.path.abspath('../test_data/state1_train_augmentation')


class BatchReader:
    """
    Have images in queue and return a batch when necessary
    """

    def __init__(self, pathin, test=False):
        """
        :param path: Path to training samples 
        """
        self.input_path = pathin
        self.test=test

        self.images = [f for f in os.listdir(os.path.join(self.input_path, 'images'))
            if not f.startswith('.')]
        shuffle(self.images)

        self.total_cnt = len(self.images)
        self.curr_cnt = 0

    def has_next_batch(self):
        """
        :return: True if there is still training sample available 
        """
        if self.test:
            # Use only one batch
            return self.curr_cnt == 0
        else:
            return self.curr_cnt < self.total_cnt

    def next_batch(self, batch_size=10):
        """
        Extract and return the next batch of input
        Return the last images if the last batch is not complete
        
        :param batch_size: Batch size
        :param test: Whether the method is called for test purpose
        :return: [batch_size, nrows, ncols], [batch_size, nrows, ncols] for samples and labels
        """
        if self.has_next_batch():
            imgs = []
            labels = []

            for i in range(batch_size):
                if self.curr_cnt < self.total_cnt:
                    image_id = self.images[self.curr_cnt]
                    label_id = image_id.split('.')[0]
                    imgs.append(
                        np.expand_dims(
                            np.load(os.path.join(self.input_path, 'images', image_id)), 0))
                    labels.append(
                        np.expand_dims(
                            np.load(os.path.join(self.input_path, 'labels', label_id + '.npy')), 0))

                    self.curr_cnt += 1

            imgs = np.concatenate(imgs, axis=0)
            labels = np.concatenate(labels, axis=0)

            if self.test:
                print("imgs.shape: ", imgs.shape)
                print("labels.shape: ", labels.shape)
            return imgs, labels
        else:
            return None

if __name__ == "__main__":
    batch_reader=BatchReader(INPUT_DIR, test=True)
    while batch_reader.has_next_batch():
        batch_reader.next_batch(10)