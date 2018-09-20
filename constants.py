import tensorflow as tf

# a small value
EPSILON = 1e-8

# here are input pipeline settings.
# you need to tweak these numbers for your system,
# it can accelerate training
SHUFFLE_BUFFER_SIZE = 15000
NUM_THREADS = 8
# read here about the buffer sizes:
# stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle

# images are resized before feeding them to the network
RESIZE_METHOD = tf.image.ResizeMethod.BILINEAR

# this is used in tf.map_fn when creating training targets or doing NMS
PARALLEL_ITERATIONS = 8

# this can be important
BATCH_NORM_MOMENTUM = 0.9
