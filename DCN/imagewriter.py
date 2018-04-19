import os
import cv2


def image_writer(path, imgs, prefix=''):
    """
    Write batch of images to .png
    
    :param path: path to images
    :param imgs: The images to be output. With shape [batch_size, nrows, ncols]
    :param prefix: prefix used for filename
    """
    try:
        assert (imgs.ndim == 3 or imgs.ndim == 2)
        if imgs.ndim == 3:
            for _ in range(imgs.shape[0]):
                cv2.imwrite(os.path.join(path, prefix + str(_) + '.png'), imgs[_])
        if imgs.ndim == 2:
            if prefix:
                cv2.imwrite(os.path.join(path, prefix + '.png'), imgs)
            else:
                cv2.imwrite(os.path.join(path, 'Noname.png'), imgs)

    except AssertionError:
        print("The batch of images for image_writer must have shape: "),
        print("[batch_size, nrows, ncols] or [nrows, ncols], ")
        print("while the dimension of the paramter is: %i " % imgs.ndim)
