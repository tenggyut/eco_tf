#encoding=utf8
import tensorflow as tf
import tensorflow.contrib as tfc

from eco_network import ECONet
import time
import sys
import os

def test_model():
    batch_size = 5
    image_size = 224
    time_step = 16
    images = tf.placeholder(tf.float32, shape=(batch_size, time_step, image_size, image_size, 3))
    features, logits = ECONet(batch_size, time_step, image_size)(images, False)

def test_decode(root_dir):
    with tf.Session() as sess:
        file_path = tf.placeholder(tf.string, [])
        frames_raw = tf.read_file(file_path)
        frames = tfc.ffmpeg.decode_video(frames_raw)
        for mp4 in os.listdir(root_dir):
            if '.mp4' in mp4:
                start = time.time()
                print('%s' % mp4)
                imgs = sess.run(frames, feed_dict={file_path:os.path.join(root_dir, mp4)})

                end = time.time()
                print('%s frame num: %d, cost: %f' % (mp4, imgs.shape[0], end - start))

def test_code():
    sess = tf.InteractiveSession()
    time_step = 16
    frame_num = tf.placeholder(tf.int32, [])
    images = tf.random_uniform([frame_num, 3], minval=0, maxval=5)
    idx = tf.range(frame_num)
    chunk_size = tf.cast(tf.ceil(frame_num / time_step), tf.int32)
    splits_size = tf.zeros([time_step - 1], tf.int32) + 1
    splits_size = splits_size * chunk_size

    remain_size = tf.zeros([1], tf.int32) + (frame_num - (chunk_size * (time_step - 1)))
    splits_size = tf.concat([splits_size, remain_size], axis = 0)
    splits = tf.split(idx, splits_size)
    selected_idx = []
    for i in range(time_step):
        random_offset = tf.random_uniform(shape=(), minval=0, maxval=splits_size[i], dtype=tf.int32) + i * chunk_size
        selected_idx.append(random_offset)

    selected_idx = tf.stack(selected_idx)
    selected_frames = tf.gather(images, selected_idx)

    a = sess.run(selected_frames, feed_dict={frame_num:123})
    print(a.shape)

if __name__ == '__main__':
    #test_model()
    test_decode(sys.argv[1])
