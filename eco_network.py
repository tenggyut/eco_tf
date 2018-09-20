import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from constants import BATCH_NORM_MOMENTUM

class ECONet:
    def __init__(self, batch_size, time_step = 16, image_size = 224, dropout = 0.5, is_training = True):
        self.is_training = is_training
        self.batch_size = batch_size
        self.time_step = time_step
        self.image_size = image_size
        if self.is_training:
            self.dropout = dropout
        else:
            self.dropout = 1.

    def __call__(self, inputs, class_num = 0, is_lite = True):
        """
        Arguments:
            images: a float tensor with shape [batch_size, time_step, height, width, 3],
                a batch of RGB images with pixel values in the range [0, 1].
        Returns:
            a list of float tensors where the ith tensor
            has shape [batch, time_step, height_i, width_i, channels_i].
        """

        def batch_norm(x):
            x = tf.layers.batch_normalization(
                x, axis=-1, center=True, scale=True,
                momentum=BATCH_NORM_MOMENTUM, epsilon=0.001,
                training=self.is_training, fused=True,
                name='batch_norm'
            )
            return x

        with tf.name_scope('standardize_input'):
            x = preprocess(inputs)

        x = tf.reshape(x, (self.batch_size * self.time_step, self.image_size, self.image_size, 3))

        x, res3d_input = self.head_2d_net(x, batch_norm)
        features = self.res_3d_net(res3d_input, batch_norm)
        if not is_lite:
            features_2d = self.bottom_2d_net(x, batch_norm)
            features = tf.concat([features, features_2d], axis=-1)

        if class_num:
            logits = slim.fully_connected(features, class_num)
        else:
            logits = None

        return features, logits

    def loss(self, logits, labels):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels, logits = logits)
        loss_avg = tf.reduce_mean(loss)
        return loss_avg

    def get_predictions(self, logits):
        pred_prob = tf.nn.softmax(logits, name = 'pred_probs')
        pred_labels = tf.argmax(pred_prob, -1, name = 'pred_labels')
        return {'pred_probs': pred_prob, 'pred_labels':pred_labels}

    def bottom_2d_net(self, inception_3c_out, bn_func):
        params = {
            'padding': 'SAME',
            'activation_fn': tf.nn.relu,
            'normalizer_fn': bn_func
        }
        with slim.arg_scope([slim.conv2d], **params):
            with tf.variable_scope('inception_4a'):
                with tf.variable_scope('branch0'):
                    x1 = slim.conv2d(inception_3c_out, 224, (1, 1), scope='conv1_1x1')
                with tf.variable_scope('branch1'):
                    x2 = slim.conv2d(inception_3c_out, 64, (1, 1), scope='conv1_1x1')
                    x2 = slim.conv2d(x2, 96, (3, 3), scope='conv2_3x3')
                with tf.variable_scope('branch2'):
                    x3 = slim.conv2d(inception_3c_out, 96, (1, 1), scope='conv1_1x1')
                    x3 = slim.conv2d(x3, 128, (3, 3), scope='conv2_3x3')
                    x3 = slim.conv2d(x3, 128, (3, 3), scope='conv3_3x3')
                with tf.variable_scope('branch3'):
                    x4 = slim.avg_pool2d(inception_3c_out, [3, 3], stride = 1, scope='pool1_3x3', padding = 'SAME')
                    x4 = slim.conv2d(x4, 128, (1, 1), scope='conv1_1x1')

                x = tf.concat([x1, x2, x3, x4], axis=-1)

            with tf.variable_scope('inception_4b'):
                with tf.variable_scope('branch0'):
                    x1 = slim.conv2d(x, 192, (1, 1), scope='conv1_1x1')
                with tf.variable_scope('branch1'):
                    x2 = slim.conv2d(x, 96, (1, 1), scope='conv1_1x1')
                    x2 = slim.conv2d(x2, 128, (3, 3), scope='conv2_3x3')
                with tf.variable_scope('branch2'):
                    x3 = slim.conv2d(x, 96, (1, 1), scope='conv1_1x1')
                    x3 = slim.conv2d(x3, 128, (3, 3), scope='conv2_3x3')
                    x3 = slim.conv2d(x3, 128, (3, 3), scope='conv3_3x3')
                with tf.variable_scope('branch3'):
                    x4 = slim.avg_pool2d(x, [3, 3], stride = 1, scope='pool1_3x3', padding = 'SAME')
                    x4 = slim.conv2d(x4, 128, (1, 1), scope='conv1_1x1')

                x = tf.concat([x1, x2, x3, x4], axis=-1)

            with tf.variable_scope('inception_4c'):
                with tf.variable_scope('branch0'):
                    x1 = slim.conv2d(x, 160, (1, 1), scope='conv1_1x1')
                with tf.variable_scope('branch1'):
                    x2 = slim.conv2d(x, 128, (1, 1), scope='conv1_1x1')
                    x2 = slim.conv2d(x2, 160, (3, 3), scope='conv2_3x3')
                with tf.variable_scope('branch2'):
                    x3 = slim.conv2d(x, 128, (1, 1), scope='conv1_1x1')
                    x3 = slim.conv2d(x3, 160, (3, 3), scope='conv2_3x3')
                    x3 = slim.conv2d(x3, 160, (3, 3), scope='conv3_3x3')
                with tf.variable_scope('branch3'):
                    x4 = slim.avg_pool2d(x, [3, 3], stride = 1, scope='pool1_3x3', padding = 'SAME')
                    x4 = slim.conv2d(x4, 128, (1, 1), scope='conv1_1x1')

                x = tf.concat([x1, x2, x3, x4], axis=-1)

            with tf.variable_scope('inception_4d'):
                with tf.variable_scope('branch0'):
                    x1 = slim.conv2d(x, 96, (1, 1), scope='conv1_1x1')
                with tf.variable_scope('branch1'):
                    x2 = slim.conv2d(x, 128, (1, 1), scope='conv1_1x1')
                    x2 = slim.conv2d(x2, 192, (3, 3), scope='conv2_3x3')
                with tf.variable_scope('branch2'):
                    x3 = slim.conv2d(x, 160, (1, 1), scope='conv1_1x1')
                    x3 = slim.conv2d(x3, 192, (3, 3), scope='conv2_3x3')
                    x3 = slim.conv2d(x3, 192, (3, 3), scope='conv3_3x3')
                with tf.variable_scope('branch3'):
                    x4 = slim.avg_pool2d(x, [3, 3], stride = 1, scope='pool1_3x3', padding = 'SAME')
                    x4 = slim.conv2d(x4, 128, (1, 1), scope='conv1_1x1')

                x = tf.concat([x1, x2, x3, x4], axis=-1)

            with tf.variable_scope('inception_4e'):

                with tf.variable_scope('branch0'):
                    x1 = slim.conv2d(x, 128, (1, 1), scope='conv1_1x1')
                    x1 = slim.conv2d(x1, 192, (3, 3), stride = 2, scope='conv2_3x3')

                with tf.variable_scope('branch1'):
                    x2 = slim.conv2d(x, 192, (1, 1), scope='conv1_1x1')
                    x2 = slim.conv2d(x2, 256, (3, 3), scope='conv2_3x3')
                    x2 = slim.conv2d(x2, 256, (3, 3), stride = 2, scope='conv3_3x3')

                with tf.variable_scope('branch2'):
                    x3 = slim.max_pool2d(x, [3, 3], stride = 2, scope='pool1_3x3', padding = 'SAME')

                x = tf.concat([x1, x2, x3], axis=-1)

            with tf.variable_scope('inception_5a'):
                with tf.variable_scope('branch0'):
                    x1 = slim.conv2d(x, 352, (1, 1), scope='conv1_1x1')
                with tf.variable_scope('branch1'):
                    x2 = slim.conv2d(x, 192, (1, 1), scope='conv1_1x1')
                    x2 = slim.conv2d(x2, 320, (3, 3), scope='conv2_3x3')
                with tf.variable_scope('branch2'):
                    x3 = slim.conv2d(x, 160, (1, 1), scope='conv1_1x1')
                    x3 = slim.conv2d(x3, 224, (3, 3), scope='conv2_3x3')
                    x3 = slim.conv2d(x3, 224, (3, 3), scope='conv3_3x3')
                with tf.variable_scope('branch3'):
                    x4 = slim.avg_pool2d(x, [3, 3], stride = 1, scope='pool1_3x3', padding = 'SAME')
                    x4 = slim.conv2d(x4, 128, (1, 1), scope='conv1_1x1')

                x = tf.concat([x1, x2, x3, x4], axis=-1)

            with tf.variable_scope('inception_5b'):
                with tf.variable_scope('branch0'):
                    x1 = slim.conv2d(x, 352, (1, 1), scope='conv1_1x1')
                with tf.variable_scope('branch1'):
                    x2 = slim.conv2d(x, 192, (1, 1), scope='conv1_1x1')
                    x2 = slim.conv2d(x2, 320, (3, 3), scope='conv2_3x3')
                with tf.variable_scope('branch2'):
                    x3 = slim.conv2d(x, 160, (1, 1), scope='conv1_1x1')
                    x3 = slim.conv2d(x3, 224, (3, 3), scope='conv2_3x3')
                    x3 = slim.conv2d(x3, 224, (3, 3), scope='conv3_3x3')
                with tf.variable_scope('branch3'):
                    x4 = slim.avg_pool2d(x, [3, 3], stride = 1, scope='pool1_3x3', padding = 'SAME')
                    x4 = slim.conv2d(x4, 128, (1, 1), scope='conv1_1x1')

                x = tf.concat([x1, x2, x3, x4], axis=-1)

            with tf.variable_scope('global_avg'):
                x = slim.avg_pool2d(x, [7, 7], stride = 1, scope='pool1_7x7')
                x = tf.nn.dropout(x, self.dropout)
                x = tf.reshape(x, (-1, 1, 16, 1024))
                x = slim.avg_pool2d(x, [1, 16], stride = 1, scope='pool1_7x7')
                x = tf.reshape(x, (-1, 1024))

        return x

    def res_3d_net(self, x, bn_func):
        params = {
            'padding': 'SAME',
            'activation_fn': tf.nn.relu,
            'normalizer_fn': bn_func
        }

        x = tf.reshape(x, (self.batch_size, self.time_step, 28, 28, 96))
        with slim.arg_scope([slim.conv3d], **params):
            with tf.variable_scope('res3a'):
                res3a_2n = slim.conv3d(x, 128, kernel_size=[3, 3, 3], stride=[1, 1, 1], scope='conv3d1')
            with tf.variable_scope('res3b_1'):
                res3b_1 = slim.conv3d(res3a_2n, 128, kernel_size=[3, 3, 3], stride=[1, 1, 1], scope='conv3d2')
            with tf.variable_scope('res3b_2'):
                res3b_2 = slim.conv3d(res3b_1, 128, kernel_size=[3, 3, 3], stride=[1, 1, 1],
                  activation_fn = None, normalizer_fn = None, scope='conv3d1')

            with tf.variable_scope('res3b'):
                res3b = tf.add(res3a_2n, res3b_2)
                res3b_bn = bn_func(res3b)
                res3b_relu = tf.nn.relu(res3b_bn)

            with tf.variable_scope('res4a_down'):
                res4a_down = slim.conv3d(res3b_relu, 256, kernel_size=[3, 3, 3], stride=[2, 2, 2],
                 activation_fn = None, normalizer_fn = None, scope='conv3d1')

            with tf.variable_scope('res4a_1'):
                res4a1 = slim.conv3d(res3b_relu, 256, kernel_size=[3, 3, 3], stride=[2, 2, 2], scope='conv3d1')
            with tf.variable_scope('res4a_2'):
                res4a2 = slim.conv3d(res4a1, 256, kernel_size=[3, 3, 3], stride=[1, 1, 1],
                 activation_fn = None, normalizer_fn = None, scope='conv3d2')
            with tf.variable_scope('res4a'):
                res4a = tf.add(res4a_down, res4a2)
                res4a_bn = bn_func(res4a)
                res4a_relu = tf.nn.relu(res4a_bn)

            with tf.variable_scope('res4b_1'):
                res4b1 = slim.conv3d(res4a_relu, 256, kernel_size=[3, 3, 3], stride=[1, 1, 1], scope='conv3d1')
            with tf.variable_scope('res4b_2'):
                res4b2 = slim.conv3d(res4b1, 256, kernel_size=[3, 3, 3], stride=[1, 1, 1],
                   activation_fn = None, normalizer_fn = None, scope='conv3d2')
            with tf.variable_scope('res4b'):
                res4b = tf.add(res4a, res4b2)
                res4b_bn = bn_func(res4b)
                res4b_relu = tf.nn.relu(res4b_bn)

            with tf.variable_scope('res5a_down'):
                res5a_down = slim.conv3d(res4b_relu, 512, kernel_size=[3, 3, 3], stride=[2, 2, 2],
                    activation_fn = None, normalizer_fn = None, scope='conv3d1')
            with tf.variable_scope('res5a_1'):
                res5a1 = slim.conv3d(res4b_relu, 512, kernel_size=[3, 3, 3], stride=[2, 2, 2], scope='conv3d1')
            with tf.variable_scope('res5a_2'):
                res5a2 = slim.conv3d(res5a1, 512, kernel_size=[3, 3, 3], stride=[1, 1, 1],
                 activation_fn = None, normalizer_fn = None, scope='conv3d2')
            with tf.variable_scope('res5a'):
                res5a = tf.add(res5a_down, res5a2)
                res5a_bn = bn_func(res5a)
                res5a_relu = tf.nn.relu(res5a_bn)

            with tf.variable_scope('res5b_1'):
                res5b1 = slim.conv3d(res5a_relu, 512, kernel_size=[3, 3, 3], stride=[1, 1, 1], scope='conv3d1')
            with tf.variable_scope('res5b_2'):
                res5b2 = slim.conv3d(res5b1, 512, kernel_size=[3, 3, 3], stride=[1, 1, 1],
                   activation_fn = None, normalizer_fn = None, scope='conv3d2')
            with tf.variable_scope('res5b'):
                res5b = tf.add(res5a, res5b2)
                res5b_bn = bn_func(res5b)
                res5b_relu = tf.nn.relu(res5b_bn)

            with tf.variable_scope('global_avg'):
                logits = slim.avg_pool3d(res5b_relu, kernel_size = [4,7,7], stride = [1,1,1])

            with tf.variable_scope('res_logits'):
                logits = slim.flatten(logits)
                logits = tf.nn.dropout(logits, self.dropout)

        return logits

    def head_2d_net(self, x, batch_norm_func):
        params = {
            'padding': 'SAME',
            'activation_fn': tf.nn.relu,
            'normalizer_fn': batch_norm_func
        }
        with slim.arg_scope([slim.conv2d], **params):
            with slim.arg_scope([slim.max_pool2d], stride=2, padding='SAME'):
                x = slim.conv2d(x, 64, (7, 7), stride=2, scope='conv1')
                x = slim.max_pool2d(x, (3, 3), scope='pool1')
                x = slim.conv2d(x, 192, (3, 3), stride=1, scope='conv2')
                x = slim.max_pool2d(x, (3, 3), scope='pool2')

            with tf.variable_scope('inception_3a'):
                with tf.variable_scope('branch0'):
                    x1 = slim.conv2d(x, 64, (1, 1), scope='conv1_1x1')

                with tf.variable_scope('branch1'):
                    x2 = slim.conv2d(x, 64, (1, 1), scope='conv1_1x1')
                    x2 = slim.conv2d(x2, 64, (3, 3), scope='conv2_3x3')

                with tf.variable_scope('branch2'):
                    x3 = slim.conv2d(x, 64, (1, 1), scope='conv1_1x1')
                    x3 = slim.conv2d(x3, 96, (3, 3), scope='conv2_3x3')
                    x3 = slim.conv2d(x3, 96, (3, 3), scope='conv3_3x3')

                with tf.variable_scope('branch3'):
                    x4 = slim.avg_pool2d(x, [3, 3], stride = 1, scope='pool1_3x3', padding = 'SAME')
                    x4 = slim.conv2d(x4, 32, (1, 1), scope='conv1_1x1')

                x = tf.concat([x1, x2, x3, x4], axis=-1)

            with tf.variable_scope('inception_3b'):
                with tf.variable_scope('branch0'):
                    x1 = slim.conv2d(x, 64, (1, 1), scope='conv1_1x1')

                with tf.variable_scope('branch1'):
                    x2 = slim.conv2d(x, 64, (1, 1), scope='conv1_1x1')
                    x2 = slim.conv2d(x2, 96, (3, 3), scope='conv2_3x3')

                with tf.variable_scope('branch2'):
                    x3 = slim.conv2d(x, 64, (1, 1), scope='conv1_1x1')
                    x3 = slim.conv2d(x3, 96, (3, 3), scope='conv2_3x3')
                    x3 = slim.conv2d(x3, 96, (3, 3), scope='conv3_3x3')

                with tf.variable_scope('branch3'):
                    x4 = slim.avg_pool2d(x, [3, 3], stride = 1, scope='pool1_3x3', padding = 'SAME')
                    x4 = slim.conv2d(x4, 64, (1, 1), scope='conv1_1x1')

                x = tf.concat([x1, x2, x3, x4], axis=-1)

            with tf.variable_scope('inception_3c'):

                with tf.variable_scope('branch0'):
                    x1 = slim.conv2d(x, 128, (1, 1), scope='conv1_1x1')
                    x1 = slim.conv2d(x1, 160, (3, 3), stride = 2, scope='conv2_3x3')

                with tf.variable_scope('branch1'):
                    x2 = slim.conv2d(x, 64, (1, 1), scope='conv1_1x1')
                    x2 = slim.conv2d(x2, 96, (3, 3), scope='conv2_3x3', activation_fn = None)
                    res3d_input = tf.identity(x2)    #batch_size * time_step, 28,28,96
                    x2 = tf.nn.relu(x2)
                    x2 = slim.conv2d(x2, 96, (3, 3), stride = 2, scope='conv3_3x3')

                with tf.variable_scope('branch2'):
                    x3 = slim.max_pool2d(x, [3, 3], stride = 2, scope='pool1_3x3', padding = 'SAME')

                x = tf.concat([x1, x2, x3], axis=-1)
            return x, res3d_input

def preprocess(images):
    """Transform images before feeding them to the network."""
    return (2.0*images) - 1.0
