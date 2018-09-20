#encoding=utf8

import os
import sys
import threading

import tensorflow as tf
import tensorflow.contrib as tfc
import numpy as np
import cv2

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('video_root', None, 'video root dir')
tf.flags.DEFINE_string('video_list', None, 'csv file contains video paths')
tf.flags.DEFINE_integer('threads', 1, 'worker thread num')
tf.flags.DEFINE_integer('samples_per_record', 1000, 'worker thread num')
tf.flags.DEFINE_string('out_dir', None, 'output dir')

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def parse_annotation(csv_file):
    res = []
    with open(csv_file) as f:
        for l in f:
            l = l.strip()
            if l:
                fields = l.split(',')
                if fields[0] == 'org_video':
                    continue
                assert(len(fields) == 2)
                anno = {}
                anno['path'] = fields[0].replace('/MTSVRC/', '')
                anno['label_idx'] = int(fields[1])
                res.append(anno)
    print('%s contains %d videos' % (csv_file, len(res)))
    return res

sess = tf.Session()
video_path_ = tf.placeholder(tf.string, [])
frames_raw = tf.read_file(video_path_)
frames_tf = tfc.ffmpeg.decode_video(frames_raw)

class TfWorker(threading.Thread):
    def __init__(self, name, annos):
        threading.Thread.__init__(self)
        self.name = name
        self.annos = annos

    def run(self):

        tf_count = 0
        writer = tf.python_io.TFRecordWriter('%s/%s_%d.tfrecord' % (FLAGS.out_dir, self.name, tf_count))
        sample_count = 0
        finish_count = 0
        total_count = len(self.annos)
        for anno in self.annos:
            if sample_count > FLAGS.samples_per_record:
                writer.close()
                tf_count += 1
                writer = tf.python_io.TFRecordWriter('%s/%s_%d.tfrecord' % (FLAGS.out_dir, self.name, tf_count))
                sample_count = 0

            video_path = os.path.join(FLAGS.video_root, anno['path'])
            video = cv2.VideoCapture(video_path)
            frame_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

            try:
                frames_tf_ = sess.run(frames_tf, feed_dict = {video_path_: video_path})
                if frames_tf_.shape[0] != frame_num or frames_tf_.shape[1] != height or frames_tf_.shape[2] != width:
                    print('invalid video_path: %s, decoded meta info is: %s, cv meta info: %d %d %d' % (video_path, str(frames_tf_.shape), frame_num, height, width))
                    continue
            except:
                print('invalid video_path: %s, decoded meta info is: %s, cv meta info: %d %d %d' % (video_path, str(frames_tf_.shape), frame_num, height, width))
                continue

            if frame_num <= 0:
                print('invalid video %s: contains 0 frames' % video_path)
                continue

            feature = {}
            feature['height'] = _int64_feature(height)
            feature['width'] = _int64_feature(width)
            feature['depth'] = _int64_feature(3)
            feature['frame_num'] = _int64_feature(frame_num)
            feature['label_idx'] = _int64_feature(anno['label_idx'])
            feature['filepath'] = _bytes_feature(video_path.encode())


            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            sample_count += 1
            finish_count += 1
            print('%s finish %d...%d to go' % (self.name, finish_count, total_count - finish_count) )


def main(unused_argv):
    annos = parse_annotation(FLAGS.video_list)

    workers = []
    worker_idx = 0
    for chunk in np.array_split(annos, FLAGS.threads):
        worker = TfWorker('kin_%d' % worker_idx, chunk)
        workers.append(worker)
        worker_idx += 1

    print('start to work....')
    for worker in workers:
        worker.start()

    for worker in workers:
        worker.join()


if __name__ == '__main__':
    tf.app.run(main)
