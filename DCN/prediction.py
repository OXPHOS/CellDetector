import tensorflow as tf
import os
import deepcn
import predict_data_wrapper
import checkpath
import numpy as np

MODEL_PATH = os.path.abspath('./logs/model.ckpt')

INPUT_DIR = os.path.abspath('../test_data/stage1_test')

OUTPUT_DIR = os.path.abspath('../test_data/stage1_prediction')


def prediction(input_path=INPUT_DIR,
               output_path=OUTPUT_DIR,
               model_path=MODEL_PATH,
               test=False):
    """
    Predict cell nuclei from input images based on saved model
    
    :param input_path: path to input images to be predicted
    :param output_path: path to output the predicted images and single cell split csv files
    :param model_path: path to saved model
    :param test: whether the function is called for test
    """

    X = tf.placeholder(shape=[None, chunk_size, chunk_size], dtype=tf.float32, name='input_area')
    y_inter = deepcn.deepcn(X, chunk_size, False)
    y_pred = tf.cast(tf.argmax(tf.squeeze(y_inter), -1), tf.uint8)

    img_ids = []
    for name in os.listdir(input_path):
        if os.path.isdir(os.path.join(input_path, name)):
            img_ids.append(name)
    all_preds = np.zeros((len(img_ids), 256, 256))
    print('num of images: ', len(img_ids))

    loader = tf.train.Saver()

    with tf.Session() as sess:
        print("Import model from: %s" %model_path)
        loader.restore(sess, model_path)
        # sess.run(tf.global_variables_initializer())

        batch_start_pos = 0
        while batch_start_pos < len(img_ids):
            batch_size = 100
            batch_end_pos = min(batch_start_pos + batch_size, len(img_ids))
            print('predict from %s, to %s' % (batch_start_pos, batch_end_pos))
            batch = img_ids[batch_start_pos:batch_end_pos]
            pw = predict_data_wrapper.PredictWrapper(path=input_path,
                                                     resize_size=chunk_size,
                                                     img_ids=batch)
            input_arr = pw.ResizedTestData()
            print("input_arr.shape: ", input_arr.shape)
            # input test_data_batch, output prediction of shape batch_size * 256 * 256
            pred_arr = sess.run(y_pred, feed_dict={X: input_arr})
            print("pred_arr.shape: ", pred_arr.shape)
            all_preds[batch_start_pos:batch_end_pos] = pred_arr
            pw.OutputPrediction(pred_arr*100, path=output_path)
            batch_start_pos = batch_end_pos

        # Use all img_ids and all_preds to generate single cell split csv file
        pw = predict_data_wrapper.PredictWrapper(path=input_path,
                                                 resize_size=chunk_size,
                                                 img_ids=img_ids)
        pw.GenerateSubmit(all_preds, output_path, cutoff=0.5)


if __name__ == '__main__':
    chunk_size = deepcn.FLAGS.window_size
    checkpath.check_path(OUTPUT_DIR)
    prediction(INPUT_DIR, OUTPUT_DIR, MODEL_PATH)