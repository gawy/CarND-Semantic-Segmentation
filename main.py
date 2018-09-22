#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from glob import glob
from tqdm import tqdm


COLLECTION_ASSERTS = 'ASSERTS'

# Check TensorFlow Version
from helper import gen_batch_function

assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    graph = tf.get_default_graph()

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    l_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    l_keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)

    l_out3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    l_out4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    l_out5 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    l_out5 = tf.stop_gradient(l_out5)

    return l_input, l_keep, l_out3, l_out4, l_out5
#tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

    l_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, 1, padding='same')
    l_upsc_1 = tf.layers.conv2d_transpose(l_1x1, num_classes, 4, 2, padding='same', name='l_conv_1x1', kernel_regularizer=regularizer)

    l_upsc_2 = tf.layers.conv2d_transpose(l_upsc_1, num_classes, 4, 2, padding='same', kernel_regularizer=regularizer)
    l_upsc_3 = tf.layers.conv2d_transpose(l_upsc_2, num_classes, 16, 8, padding='same', name='l_ups_last', kernel_regularizer=regularizer)

    #TODO: add skip connections and play with model structure

    return l_upsc_3
#tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    l_logits = tf.reshape(nn_last_layer, [-1, num_classes])
    loss_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=nn_last_layer, labels=correct_label)

    tf.Print(loss_cross_entropy, [tf.losses.get_regularization_losses()], message="L2 losses: ")
    loss = tf.add(tf.reduce_mean(loss_cross_entropy), sum(tf.losses.get_regularization_losses()))

    l_prediction_indexes = tf.argmax(tf.nn.softmax(l_logits), axis=1)
    l_predictions = tf.one_hot(l_prediction_indexes, num_classes)  # convert last dimention to 1-hot

    # tf.add_to_collection(COLLECTION_ASSERTS, tf.Print(l_predictions, [tf.shape(l_predictions)], message='Prediction 1-hot size: '))
    # tf.add_to_collection(COLLECTION_ASSERTS, tf.assert_equal(tf.shape(l_predictions), [-1, 2], message="Predictions 1_hot shape: ", name="pred-1hot-size"))
    tests._assert_tensor_shape(l_predictions, [None, 2], "Predictions 1_hot shape")

    l_label_1_hot = tf.one_hot(tf.argmax(tf.reshape(correct_label, (-1, num_classes)), axis=1), num_classes)
    # tf.add_to_collection(COLLECTION_ASSERTS, tf.Assert(l_label_1_hot.get_shape() == (-1, 3), data=["Labels 1-hot shape: "], name="label-1hot-size"))
    tests._assert_tensor_shape(l_label_1_hot, [None, 2], "Labels 1_hot shape")

    accuracy, accuracy_op = tf.metrics.mean_iou(l_label_1_hot, l_predictions, num_classes, name="accuracy_iou")

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return l_logits, train_op, loss, accuracy, accuracy_op
#tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, data_folder='', accuracy=None, accuracy_update=None):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
    batch_count = int(len(image_paths) / batch_size) + 1

    print("Started training")
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    run_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='accuracy_iou')
    acc_initializer = tf.variables_initializer(run_vars)
    saver = tf.train.Saver()
    print("Variables initialized")

    asserts = tf.get_collection(COLLECTION_ASSERTS)
    with tf.control_dependencies(asserts) :

        for epoch in range(epochs):
            batches_pbar = tqdm(total=batch_count, desc='Epoch {:>2}/{}'.format(epoch+1, epochs), unit='batch')
            acc_initializer.run()

            for batch in get_batches_fn(batch_size):
                hist = sess.run([train_op, cross_entropy_loss, accuracy_update], feed_dict={input_image: batch[0], correct_label: batch[1], keep_prob: 0.5})
                batches_pbar.update()

                batches_pbar.write("Loss={:.2f}".format(hist[1]))

            accu = sess.run([accuracy])
            print("Epoch ended. Loss={:.2f}, \n accuracy={:.2f}".format(hist[1], accu[0]))
            batches_pbar.close()

    # saver.save(sess, './runs/model.ckpt')
    pass
#tests.test_train_nn(train_nn)


def run():
    epochs = 10
    batch_size = 20
    learn_rate = 1e-2

    num_classes = 2
    image_shape = (160, 576)
    label_channels = 2
    data_dir = '/data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:

        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        images_path = os.path.join(data_dir, 'data_road/training')
        get_batches_fn = helper.gen_batch_function(images_path, image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        l_input, l_keep, l_out3, l_out4, l_out7  = load_vgg(sess, vgg_path)
        l_out = layers(l_out3, l_out4, l_out7, num_classes)

        l_label = tf.placeholder(tf.float32, (None, image_shape[0], image_shape[1], label_channels), name='Label')
        l_logits, train_op, loss_op, accuracy, accuracy_op = optimize(l_out, l_label, learn_rate, num_classes)


        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, loss_op, l_input, l_label, l_keep,
                 learn_rate, images_path, accuracy=accuracy, accuracy_update=accuracy_op)

        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, l_logits, l_keep, l_input)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
