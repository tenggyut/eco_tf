#! /usr/bin/env bash

train_out_dir=/home/ubuntu/disk_e/dataset/sfn_dataset/train_shards
val_out_dir=/home/ubuntu/disk_e/dataset/sfn_dataset/val_shards


rm -rf $train_out_dir
rm -rf $val_out_dir
mkdir $train_out_dir
mkdir $val_out_dir

python create_tfrecords.py --out_dir $train_out_dir --samples_per_record 50 --video_list ~/disk_e/dataset/sfn_dataset/train_list.txt --video_root ~/disk_e/dataset/sfn_dataset/ --threads 2
python create_tfrecords.py --out_dir $val_out_dir --samples_per_record 20 --video_list ~/disk_e/dataset/sfn_dataset/val_list.txt --video_root ~/disk_e/dataset/sfn_dataset/ --threads 2

