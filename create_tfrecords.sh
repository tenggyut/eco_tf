#! /usr/bin/env bash

train_out_dir=/home/ubuntu/disk_e/kinetics400_shards/train
val_out_dir=/home/ubuntu/disk_e/kinetics400_shards/val


rm -rf $train_out_dir
rm -rf $val_out_dir
mkdir $train_out_dir
mkdir $val_out_dir

python create_tfrecords.py --out_dir $train_out_dir --samples_per_record 1000 --video_list ~/disk_a/DongAn/video-understanding/data/compress/kinetics_train_1.csv --video_root ~/disk_a/DongAn/video-understanding/ --threads 10
python create_tfrecords.py --out_dir $val_out_dir --samples_per_record 1000 --video_list ~/disk_a/DongAn/video-understanding/data/compress/kinetics_val_1.csv --video_root ~/disk_a/DongAn/video-understanding/ --threads 10

