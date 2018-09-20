import tensorflow as tf
import json
import os
import sys

from model import model_fn
from pipeline import Pipeline
tf.logging.set_verbosity('INFO')

if len(sys.argv) == 2:
    CONFIG = sys.argv[1]
else:
    CONFIG = 'config.json'

params = json.load(open(CONFIG))

label_map = []
with open(params['label_path']) as f:
    for l in f:
        l = l.strip()
        if l:
            label_map.append(l)
params['class_num'] = len(label_map)

def get_input_fn(is_training=True):

    image_size = params['image_size']
    # (for evaluation i use images of different sizes)
    dataset_path = params['train_dataset'] if is_training else params['val_dataset']
    batch_size = params['batch_size']
    time_step = params['time_step']

    # for evaluation it's important to set batch_size to 1

    filenames = os.listdir(dataset_path)
    filenames = [n for n in filenames if n.endswith('.tfrecord')]
    filenames = [os.path.join(dataset_path, n) for n in sorted(filenames)]

    def input_fn():
        with tf.device('/cpu:0'), tf.name_scope('input_pipeline'):
            pipeline = Pipeline(
                filenames,
                batch_size=batch_size, image_size=image_size,time_step=time_step,
                repeat=is_training, shuffle=is_training,
                augmentation=is_training
            )
            features, labels = pipeline.get_batch()
        return features, labels

    return input_fn


config = tf.ConfigProto()

run_config = tf.estimator.RunConfig()
run_config = run_config.replace(
    model_dir=params['model_dir'],
    session_config=config,
    save_summary_steps=200,
    save_checkpoints_secs=600,
    log_step_count_steps=100
)


train_input_fn = get_input_fn(is_training=True)
val_input_fn = get_input_fn(is_training=False)
estimator = tf.estimator.Estimator(model_fn, params=params, config=run_config)

train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=params['num_steps'])
eval_spec = tf.estimator.EvalSpec(val_input_fn, steps=None, start_delay_secs=1800, throttle_secs=1800)
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
