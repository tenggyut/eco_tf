import os
import tensorflow as tf

from eco_network import ECONet
from i3d import InceptionI3d

def model_fn(features, labels, mode, params, config):
    # the base network
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    batch_size = params['batch_size']

    if params['net'] == 'eco':
        net = ECONet(batch_size, params['time_step'], is_training = is_training)
        features_, logits = net(features['images'], class_num = params['class_num'], is_lite = False)
        predictions = net.get_predictions(logits)
    elif params['net'] == 'i3d':
        net = InceptionI3d(params['class_num'], spatial_squeeze=True, final_endpoint='Mixed_5c')
        rgb_logits, predictions = net.get_finetunning(features['images'], params['pretrain_ckpt_path'],
         is_training=is_training, dropout_keep_prob=params['dropout_keep_prob'])

    if mode == tf.estimator.ModeKeys.PREDICT:
        # this is required for exporting a savedmodel
        export_outputs = tf.estimator.export.PredictOutput({
            name: tf.identity(tensor, name)
            for name, tensor in predictions.items()
        })
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions,
            export_outputs={'outputs': export_outputs}
        )

    # add L2 regularization
    with tf.name_scope('weight_decay'):
        add_weight_decay(params['weight_decay'])
        regularization_loss = tf.losses.get_regularization_loss()

    # create localization and classification losses
    losses = net.loss(logits, labels['labels'])
    tf.losses.add_loss(losses)
    tf.summary.scalar('regularization_loss', regularization_loss)
    tf.summary.scalar('classification_loss', losses)
    total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

    if mode == tf.estimator.ModeKeys.EVAL:

        #batch_size = features['images'].shape.as_list()[0]
        #assert batch_size == 1

        with tf.name_scope('evaluator'):
            eval_metric_ops = {'acc': tf.metrics.accuracy(labels['labels'], predictions['pred_labels'])}

        return tf.estimator.EstimatorSpec(
            mode, loss=total_loss,
            eval_metric_ops=eval_metric_ops
        )

    assert mode == tf.estimator.ModeKeys.TRAIN
    with tf.variable_scope('learning_rate'):
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.piecewise_constant(global_step, params['lr_boundaries'], params['lr_values'])
        tf.summary.scalar('learning_rate', learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops), tf.variable_scope('optimizer'):
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9, use_nesterov=True)
        grads_and_vars = optimizer.compute_gradients(total_loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step)

    for g, v in grads_and_vars:
        if g is not None:
            tf.summary.histogram(v.name[:-2] + '_hist', v)
            tf.summary.histogram(v.name[:-2] + '_grad_hist', g)
        else:
            print(v)

    return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)


def add_weight_decay(weight_decay):
    """Add L2 regularization to all (or some) trainable kernel weights."""
    weight_decay = tf.constant(
        weight_decay, tf.float32,
        [], 'weight_decay'
    )
    trainable_vars = tf.trainable_variables()
    kernels = [v for v in trainable_vars if 'weights' in v.name]
    for K in kernels:
        x = tf.multiply(weight_decay, tf.nn.l2_loss(K))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, x)
