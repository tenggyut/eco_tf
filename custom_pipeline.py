#encoding=utf8
import os
import time
import random
import sys
import multiprocessing

import numpy as np
import cv2
import utils

def eco_preprocessing(frames, batch_size, time_step, image_size, input_channels, is_training):

    res = []
    selected_frames = []

    chunks = np.array_split(frames, time_step)

    for c in chunks:

        selected_idx = np.random.randint(c.shape[0])
        frame_sample = c[selected_idx]
        selected_frames.append(frame_sample)

    is_flip = random.randint(1, 10) % 2 == 0

    for f in selected_frames:
        if is_training:
            resized = utils.random_crop(f, image_size[0], image_size[1])
            if is_flip:
                resized = np.fliplr(resized)
        else:
            resized = utils.crop_center(f, image_size[0], image_size[1])

        resized = resized - [104,117,123]
        resized = resized.astype(np.float32)
        res.append(resized)

    return np.array(res)

class VideoInputPipeline:

    def __init__(self, video_root, batch_size, time_step, image_size, sample_list_file, input_channels = 3, one_sample_preprocssor = None,
     sample_enqueue_worker = 1, batch_enqueue_worker = 1, is_training = True, training_steps = 1000,
     sample_queue_buffer = 10000, batch_queue_buffer = 1000):
        self.samples = []
        with open(sample_list_file) as f:
            for l in f:
                l = l.strip()
                if l:
                    fields = l.split(',')
                    self.samples.append({'path': os.path.join(video_root, fields[0].replace('/MTSVRC/', '')), 'label_idx': fields[1]})

        np.random.shuffle(self.samples)
        self.manager = multiprocessing.Manager()

        self.sample_queue = self.manager.Queue(sample_queue_buffer)
        self.batch_queue = self.manager.Queue(batch_queue_buffer)
        self.sample_pool = multiprocessing.Pool(sample_enqueue_worker)
        self.batch_pool = multiprocessing.Pool(batch_enqueue_worker)
        self.wait_sec = 2
        self.max_steps = training_steps

        self.batch_size = batch_size
        self.time_step = time_step
        self.image_size = image_size
        self.input_channels = input_channels
        self.preprocessor = one_sample_preprocssor
        self.is_training = is_training

    @staticmethod
    def preprocessing(one_sample_batch, batch_size, time_step, image_size,
     input_channels, preprocessor, is_training):
        samples = []
        labels = []
        for one_sample in one_sample_batch:

            labels.append(one_sample['label_idx'])
            video = cv2.VideoCapture(one_sample['path'])
            frame_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            frames = []
            for _ in range(frame_num):
                ok, frame = video.read()
                if ok:
                    frames.append(frame)

            if preprocessor:
                frames = preprocessor(frames, batch_size = batch_size, time_step = time_step, image_size = image_size,
                 input_channels = input_channels, is_training = is_training)
            samples.append(frames)

        samples = np.array(samples)
        labels = np.array(labels)
        assert(samples.shape[0] == batch_size)
        assert(samples.shape[1] == time_step)
        assert(samples.shape[2] == image_size[0])
        assert(samples.shape[3] == image_size[1])
        assert(samples.shape[4] == input_channels)
        assert(labels.shape[0] == batch_size)
        return samples, labels

    @staticmethod
    def enqueue_samples(samples, sample_queue, wait_sec = 2):
        while True:
            for s in samples:
                sample_queue.put(s)


    @staticmethod
    def enqueue_batch(sample_queue, batch_queue, batch_size, time_step, image_size,
     input_channels, preprocessor, is_training, wait_sec = 2):
        while True:

            one_sample_batch = []

            while len(one_sample_batch) != batch_size:

                if sample_queue.empty():
                    time.sleep(wait_sec)
                    continue

                s = sample_queue.get()

                one_sample_batch.append(s)

            one_batch = VideoInputPipeline.preprocessing(one_sample_batch, batch_size, time_step, image_size, input_channels, preprocessor, is_training)


            batch_queue.put(one_batch)

    def start_worker(self):
        self.sample_pool.apply_async(VideoInputPipeline.enqueue_samples, args=(self.samples, self.sample_queue))
        self.batch_pool.apply_async(VideoInputPipeline.enqueue_batch, args=(self.sample_queue, self.batch_queue,
         self.batch_size, self.time_step, self.image_size, self.input_channels, self.preprocessor, self.is_training))
        print('finish starting workers......%d'%len(self.samples))

    def next_batch(self):
        for i in range(self.max_steps):
            while self.batch_queue.empty():
                print('sample queue: %d' % self.sample_queue.qsize())
                print('batch queue: %d' % self.batch_queue.qsize())
                time.sleep(self.wait_sec)

            one_batch = self.batch_queue.get()
            yield one_batch

    def close(self):
        self.sample_pool.close()
        self.batch_pool.close()
        self.sample_pool.join()
        self.batch_pool.join()


if __name__ == '__main__':
    batch_size = 3
    time_step = 16
    image_size = (224,224)
    steps = 1000
    sample_list_file = sys.argv[1]
    video_root = sys.argv[2]
    input_pipeline = VideoInputPipeline(video_root, batch_size, time_step, image_size, sample_list_file,
     training_steps = steps, one_sample_preprocssor = eco_preprocessing)
    input_pipeline.start_worker()
    for samples, labels in input_pipeline.next_batch():
        print(samples.shape, labels.shape)

    input_pipeline.close()
