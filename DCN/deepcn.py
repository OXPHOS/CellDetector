"""
Run instruction on server: 

cd workspace/oxphos/CellDetector/DCN
python deepcn.py
"""

import tensorflow as tf
import numpy as np
import os
import imageparser, imagewriter, checkpath
import BatchReader

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('logdir', os.path.abspath('./logs/run1'), 'Dir to save logs.')
flags.DEFINE_string('ckptdir', os.path.abspath('./logs/model.ckpt'), 'Path to save checkpoints.')
flags.DEFINE_string('inputdir', os.path.abspath('../test_data/stage1_train/'), 'Dir to training samples.')
flags.DEFINE_string('augdir', os.path.abspath('../test_data/stage1_train_augmentation/'),
                    'Dir to augmented training samples.')
flags.DEFINE_string('outputdir', os.path.abspath('./logs/stage1_validation'), 'Dir to output images.')

flags.DEFINE_integer('window_size', 256, 'Size of sliding window.')
flags.DEFINE_integer('batch_size', 10, 'Batch size.')

flags.DEFINE_boolean('restore_checkpoint', False, 'Whether to restore session from checkpoint')
flags.DEFINE_boolean('test_import', False, 'Whether to test image reader module.')

print("logs are saved in: ", FLAGS.logdir)
print(tf.VERSION)


def image_summary(tensor, suffix=''):
    """
    Generate images for TensorBoard from tensors.
    Tensors have dimensions [batch_size, dim, dim, 3]
     
    :param tensor: The tensor to be converted to image. Usually y_conv(predicted y)
    :param suffix: suffix for image name
    :return: None
    """
    if len(tensor.shape) != 4 and tensor.shape[-1] != 3:
        print("Unable to convert the matrix to image.")
        raise Exception
    squash = tf.cast(tf.expand_dims(tf.argmax(tensor, -1)*100, -1), tf.uint8)
    tf.summary.image('predicted%s'%suffix, squash, 10)


def weight_variable(shape):
    """
    weight_variable generates a weight variable of a given shape.

    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name="vars")


def bias_variable(shape):
    """
    bias_variable generates a bias variable of a given shape.
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def convolution(x, W, b, training=True):
    """
    Convolution layer
    
    :param x: input tensor
    :param W: weight matrix
    :param b: bias
    :return: result after convolution, batch normalization and activation
    """
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    # Batch normalization
    mean, var = tf.nn.moments(conv, [0])
    scale = tf.Variable(tf.ones(b.shape))
    bn = tf.nn.batch_normalization(conv, mean, var, scale, b, 1e-7)
    return tf.nn.relu(bn)


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')


def up_convolution(tensor, dim, W, b, _):
    """
    Up sampling.
    
    :param tensor: input to perform transpose convolution
    :param dim:
    :param W: weight matrix
    :param b: bias
    :param _: layer
    :return: classifier layer
    """
    # x = tf.image.resize_bilinear(tensor, [dim, dim])
    x = tensor

    # Generate output shape for transpose convolution
    batch_size = tf.shape(tensor)[0]
    deconv_shape = tf.stack([batch_size, dim, dim, 3])

    # Up sampling
    # output.shape = [batch_size, dim, dim, 3]
    with tf.name_scope('transpose_conv'):
        x = tf.nn.relu(tf.nn.conv2d_transpose(x, W[0], output_shape=
        deconv_shape, strides=[1, 2**(_+1), 2**(_+1), 1], padding='SAME') + b[0])

    # Convolution with kernel 3x3
    # output.shape = [batch_size, dim, dim, 3]
    with tf.name_scope('upsampling_conv_1'):
        x = convolution(x, W[1], b[1])

    # Convolution with kernel 1x1
    # output.shape = [batch_size, dim, dim, 3]
    with tf.name_scope('upsampling_conv_2'):
        x = convolution(x, W[2], b[2])

    return x


def deepcn(x, dim, train=True):
    """
    Deep contextual network
    
    :param x: input tensor, with shape [batch_size, dim, dim]
    :param dim: size of the sliding window
    :param train: whether the input is used for training. To toggle dropout layers
    :return: 3 auxiliary classifier layer and 1 fused classifier layer 
    """

    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, dim, dim, 1])
    tf.summary.image('input', x_image, 4)

    # Refer to Figure 2 in Chen. et al. for layer number information
    with tf.name_scope('initialize'):
        W_down = [[weight_variable([3, 3, 1, 64]),
             weight_variable([3, 3, 64, 64])],
             [weight_variable([3, 3, 64, 128]),
             weight_variable([3, 3, 128, 128])],
             [weight_variable([3, 3, 128, 256]),
             weight_variable([3, 3, 256, 256])]]
        tf.summary.histogram("Down sampling weight matrix 1.1", W_down[0][0])
        tf.summary.histogram("Down sampling weight matrix 2.1", W_down[1][0])
        tf.summary.histogram("Down sampling weight matrix 3.1", W_down[2][0])

        b_down = [[bias_variable([64]),
             bias_variable([64])],
             [bias_variable([128]),
             bias_variable([128])],
             [bias_variable([256]),
             bias_variable([256])]]
        tf.summary.histogram("Down sampling bias 1.1", b_down[0][0])
        tf.summary.histogram("Down sampling bias 2.1", b_down[1][0])
        tf.summary.histogram("Down sampling bias 3.1", b_down[2][0])

        W_up = [[weight_variable([3, 3, 3, 128*2**i]),
                 weight_variable([3, 3, 3, 3]),
                 weight_variable([1, 1, 3, 3])] for i in range(2)]
        tf.summary.histogram("Up sampling weight matrix 1.1", W_up[0][0])

        b_up = [[bias_variable([3]),
                bias_variable([3]),
                bias_variable([3])] for _ in range(2)]
        tf.summary.histogram("Up sampling bias 2.1", b_up[1][0])

    with tf.name_scope('layer1'):
        with tf.name_scope('conv_1_1'):
            h_conv_1_1 = convolution(x_image, W_down[0][0], b_down[0][0], train)

        if train:
            with tf.name_scope('dropout_1_1'):
                h_conv_1_1 = tf.nn.dropout(h_conv_1_1, keep_prob=0.9)

        with tf.name_scope('conv_1_2'):
            h_conv_1_2 = convolution(h_conv_1_1, W_down[0][1], b_down[0][1], train)

        # h_pool_1.shape = [batch_size, dim//2, dim//2, 64]
        with tf.name_scope('maxpool_1'):
            h_pool_1 = max_pool(h_conv_1_2)

    with tf.name_scope('layer2'):
        with tf.name_scope('conv_2_1'):
            h_conv_2_1 = convolution(h_pool_1, W_down[1][0], b_down[1][0], train)

        if train:
            with tf.name_scope('dropout_2_1'):
                h_conv_2_1 = tf.nn.dropout(h_conv_2_1, keep_prob=0.9)

        with tf.name_scope('conv_2_2'):
            h_conv_2_2 = convolution(h_conv_2_1, W_down[1][1], b_down[1][1], train)

        # h_pool_2.shape = [batch_size, dim//4, dim//4, 128]
        with tf.name_scope('maxpool_2'):
            h_pool_2 = max_pool(h_conv_2_2)

    with tf.name_scope('layer3'):
        with tf.name_scope('conv_3_1'):
            h_conv_3_1 = convolution(h_pool_2, W_down[2][0], b_down[2][0], train)

        if train:
            with tf.name_scope('dropout_3_1'):
                h_conv_3_1 = tf.nn.dropout(h_conv_3_1, keep_prob=0.9)

        with tf.name_scope('conv_3_2'):
            h_conv_3_2 = convolution(h_conv_3_1, W_down[2][1], b_down[2][1], train)

        # h_pool_3.shape = [batch_size, dim//8, dim//8, 256]
        with tf.name_scope('maxpool_3'):
            h_pool_3 = max_pool(h_conv_3_2)

    # Up-sampling
    with tf.name_scope('upconv_2'):
        h_final_2 = up_convolution(h_conv_2_2, dim, W_up[0], b_up[0], 0)
        # image_summary(h_final_2)

    with tf.name_scope('upconv_3'):
        h_final_3 = up_convolution(h_conv_3_2, dim, W_up[1], b_up[1], 1)
        # image_summary(h_final_3)

    with tf.name_scope('fusion'):
        # Fuse auxiliary classifiers

        # h_final.shape = [batch_size, dim, dim, 3]
        h_final = tf.add(h_final_2, h_final_3)
        # image_summary(h_final)

    if train:
        return [h_final_2, h_final_3, h_final]
    else:
        return h_final


def loss_calculation(y_, y):
    """
    Generic loss function, cross entropy
    :param y_: label
    :param y: prediction
    :return: loss
    """
    class_weights = tf.constant([1., 2., 2.])
    weights = tf.reduce_sum(class_weights * y_, axis=-1)
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_, logits=y)
    weighted_losses = unweighted_losses * weights
    return tf.reduce_mean(weighted_losses)


def main():
    window_size = FLAGS.window_size

    X = tf.placeholder(shape=[None, window_size, window_size], dtype=tf.float32, name='input_area')
    y = tf.placeholder(shape=[None, window_size, window_size], dtype=tf.int32, name='label')
    y_onehot = tf.stop_gradient(tf.one_hot(y, 3))

    # 3 auxiliary layers and 1 fused layer
    _C2, _C3, y_conv = deepcn(X, window_size)
    tf.summary.histogram('y_conv', y_conv)

    with tf.name_scope('y_boundary'):
        y_boundary = tf.argmax(y_conv, -1)
    tf.summary.histogram('y_boundary', y_boundary)

    with tf.name_scope('loss'):  # TODO: auxiliary classifier
        loss = sum(map(lambda y:loss_calculation(y_onehot, y), [_C2, _C3, y_conv])) / 3.
        # loss = loss_calculation(y_onehot, y_conv)
    tf.summary.scalar('loss', loss)

    with tf.name_scope('accuracy'):  # TODO: use IoU
        correct_prediction = tf.equal(tf.argmax(y_conv, -1), tf.argmax(y_onehot, -1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        image_summary(y_conv, '_prediction')
        image_summary(y_onehot, '_input')
    tf.summary.scalar('accuracy', accuracy)

    with tf.name_scope('adam_optimizer'):
        train_op = tf.train.AdamOptimizer().minimize(loss)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.logdir + '/train')  # logdir path
    train_writer.add_graph(tf.get_default_graph())  # output to 'graph' tab
    test_writer = tf.summary.FileWriter(FLAGS.logdir + '/test')
    saver = tf.train.Saver()  # Saver has to come after writer, or the graph is screwed!!!

    with tf.Session() as sess:
        if FLAGS.restore_checkpoint and tf.train.checkpoint_exists(FLAGS.ckptdir):
            print("Session restored from: %s", FLAGS.ckptdir)
            saver.restore(sess, FLAGS.ckptdir)
        else:
            sess.run(tf.global_variables_initializer())

        training_steps = 0
        image_count = 0

        imageparser.process_image(FLAGS.inputdir, FLAGS.augdir, FLAGS.window_size, test=FLAGS.test_import)
        for j in range(12):
            batch_reader = BatchReader.BatchReader(FLAGS.augdir, test=FLAGS.test_import)

            while batch_reader.has_next_batch():

                X_feeders, y_feeders = batch_reader.next_batch(FLAGS.batch_size)
                training_steps += 1

                # Validation
                if training_steps % 1000 == 0:
                    summary, _ = sess.run([merged, accuracy], feed_dict={X: X_feeders, y: y_feeders})
                    test_writer.add_summary(summary, training_steps)

                    # Export predicted image
                    content = y_boundary.eval(
                        feed_dict={X: X_feeders})*100
                    prediction = np.concatenate([X_feeders, content], 2)
                    for _ in range(X_feeders.shape[0]):
                        imagewriter.image_writer(FLAGS.outputdir, prediction[_], str(image_count))
                        image_count += 1

                # Write to TB
                if training_steps % 400 == 0:
                    print("Training_steps: %i" %training_steps)
                    summary, _ = sess.run([merged, train_op], feed_dict={X: X_feeders, y: y_feeders})
                    train_writer.add_summary(summary, training_steps)
                else:
                    sess.run(train_op, feed_dict={X: X_feeders, y: y_feeders})

                if training_steps % 5000 == 0:
                    saver.save(sess, FLAGS.ckptdir, training_steps)

        saver.save(sess, FLAGS.ckptdir)


if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    tf.set_random_seed(seed)

    checkpath.check_path(FLAGS.logdir)
    checkpath.check_path(FLAGS.outputdir)

    main()