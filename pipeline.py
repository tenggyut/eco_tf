#encoding=utf8
import tensorflow as tf
import tensorflow.contrib as tfc

from constants import SHUFFLE_BUFFER_SIZE, NUM_THREADS, RESIZE_METHOD

class Pipeline:
    """Input pipeline for training or evaluating object detectors."""

    def __init__(self, filenames, batch_size, image_size, time_step,
                 repeat=False, shuffle=False, augmentation=False):
        """
        Note: when evaluating set batch_size to 1.

        Arguments:
            filenames: a list of strings, paths to tfrecords files.
            batch_size: an integer.
            image_size: a list with two integers [width, height]
            repeat: a boolean, whether repeat indefinitely.
            shuffle: whether to shuffle the dataset.
            augmentation: whether to do data augmentation.
        """
        self.image_width = image_size[0]
        self.image_height = image_size[1]
        self.resize = False

        self.augmentation = augmentation
        self.batch_size = batch_size
        self.time_step = time_step

        def get_num_samples(filename):
            return sum(1 for _ in tf.python_io.tf_record_iterator(filename))

        num_examples = 0
        for filename in filenames:
            num_examples_in_file = get_num_samples(filename)
            assert num_examples_in_file > 0
            num_examples += num_examples_in_file
        self.num_examples = num_examples
        assert self.num_examples > 0

        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        num_shards = len(filenames)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=num_shards)

        dataset = dataset.flat_map(tf.data.TFRecordDataset)
        dataset = dataset.prefetch(buffer_size=batch_size)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
        dataset = dataset.repeat(None if repeat else 1)
        dataset = dataset.map(self._parse_and_preprocess, num_parallel_calls=NUM_THREADS)

        # we need batches of fixed size
        padded_shapes = ([self.time_step, self.image_height, self.image_width, 3], [])
        dataset = dataset.apply(
           tf.contrib.data.padded_batch_and_drop_remainder(batch_size, padded_shapes)
        )
        dataset = dataset.prefetch(buffer_size=batch_size)

        self.iterator = dataset.make_one_shot_iterator()

    def get_batch(self):
        """
        Returns:
            features: a dict with the following keys
                'images': a float tensor with shape [batch_size, time_step, image_height, image_width, 3].
                'filenames': a string tensor with shape [batch_size].
            labels: a dict with the following keys
                'labels': an int tensor with shape [batch_size].
        """
        images, labels = self.iterator.get_next()
        features = {'images': images}
        labels = {'labels': labels}
        return features, labels

    def _parse_and_preprocess(self, example_proto):
        features = dict()
        features["label_idx"] = tf.FixedLenFeature((), tf.int64)
        features["height"] = tf.FixedLenFeature((), tf.int64)
        features["width"] = tf.FixedLenFeature((), tf.int64)
        features["depth"] = tf.FixedLenFeature((), tf.int64)
        features["filepath"] = tf.FixedLenFeature([], tf.string)
        features["frame_num"] = tf.FixedLenFeature((), tf.int64)

        parsed_features = tf.parse_single_example(example_proto, features)

        def decode_video(file_path):
            video = cv2.VideoCapture(file_path)
            frames = []
            frame_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            for _ in range(frame_num):
                ok, f = video.read()
                if ok:
                    frames.append(f)
            return np.array(frames)

        frames_raw = tf.read_file(parsed_features["filepath"])
        frames = tfc.ffmpeg.decode_video(frames_raw)
        frame_num = tf.cast(parsed_features['frame_num'], tf.int32)

        chunk_size_tmp = tf.cast(tf.floor(frame_num / self.time_step), tf.int32)
        frames = tf.cond(tf.equal(chunk_size_tmp,0), lambda: tf.tile(frames, [5,1,1,1]), lambda: frames)
        frame_num = tf.cond(tf.equal(chunk_size_tmp,0), lambda: frame_num * 5, lambda: frame_num)
        chunk_size = tf.cast(tf.floor(frame_num / self.time_step), tf.int32)


        idx = tf.range(frame_num)

        splits_size = tf.zeros([self.time_step - 1], tf.int32) + 1
        splits_size = splits_size * chunk_size

        remain_size = tf.zeros([1], tf.int32) + (frame_num - (chunk_size * (self.time_step - 1)))
        splits_size = tf.concat([splits_size, remain_size], axis = 0)
        selected_idx = []
        for i in range(self.time_step):
            random_offset = tf.random_uniform(shape=(), minval=0, maxval=splits_size[i], dtype=tf.int32) + i * chunk_size
            selected_idx.append(random_offset)

        selected_idx = tf.stack(selected_idx)

        label  = tf.cast(parsed_features["label_idx"], tf.int64)
        selected_frames = tf.gather(frames, selected_idx)

        if self.augmentation:
            selected_frames = self._augmentation_fn(selected_frames)
        else:
            selected_frames = tf.image.resize_images(selected_frames, [self.image_height, self.image_width], method=RESIZE_METHOD)
            selected_frames = tf.map_fn(lambda i: tf.image.convert_image_dtype(i, tf.float32), selected_frames, dtype=tf.float32)

        return selected_frames, label

    def _augmentation_fn(self, images):
        def rescale_pixel_range(img, out_range=(-1, 1)):
            img = tf.image.convert_image_dtype(img, tf.float32)
            #img = (img * 2) - 1
            return img

        images = tf.map_fn(rescale_pixel_range, images, dtype=tf.float32)
        #images = tf.image.resize_images(images, [self.image_height, self.image_width], method=RESIZE_METHOD)
        images = tf.image.resize_image_with_pad(images, 256, 256)
        images = tf.map_fn(lambda i: tf.random_crop(i, (self.image_height, self.image_width, 3)), images)
        do_flip = tf.less(tf.random_uniform([]), 0.5)
        images = tf.cond(do_flip, lambda: tf.map_fn(lambda i: tf.image.flip_left_right(i), images), lambda: images)
        return images
