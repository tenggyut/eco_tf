#encoding=utf8
import os
import sys
import random
import numpy as np

def get_video_paths(video_root):
    res = {}
    if video_root.endswith('/'):
        video_root = video_root[:len(video_root)-1]
    for root, dirs, files in os.walk(video_root):
      for name in files:
          abs_path = os.path.join(root, name)
          label_name = abs_path.split('/')[-2]
          relative_path = abs_path.replace(video_root + '/', '')
          if name.endswith((".mp4", ".avi", ".mpg")):
              if label_name not in res:
                  res[label_name] = []
              res[label_name].append(relative_path)
    sample_count_per_class = min([len(res[label_name]) for label_name in res.keys()])
    total_count = 0
    for label_name in res.keys():
        res[label_name] = np.random.choice(res[label_name], sample_count_per_class).tolist()
        total_count += len(res[label_name])

    print("%s contains %d videos" % (video_root, total_count))
    return res

def split_datasets(video_root, out_dir, label_file, ratio = 0.7):
    dataset = get_video_paths(video_root)
    train_set = []
    val_set = []
    label_map = []
    with open(label_file) as f:
        for l in f:
            l = l.strip()
            if l:
                label_map.append(l)

    for label_name in dataset.keys():
        split_point = int(len(dataset[label_name]) * ratio)
        train_set.extend(dataset[label_name][:split_point])
        val_set.extend(dataset[label_name][split_point:])

    random.shuffle(train_set)
    random.shuffle(val_set)

    print('train_set: %d, val_set: %d' % (len(train_set), len(val_set)))
    with open(os.path.join(out_dir, 'train_list.txt'), 'w') as train_file:
        for v in train_set:
            label_idx = label_map.index(v.split('/')[-2])
            train_file.write(v + ',' + str(label_idx))
            train_file.write('\n')

    with open(os.path.join(out_dir, 'val_list.txt'), 'w') as val_file:
        for v in val_set:
            label_idx = label_map.index(v.split('/')[-2])
            val_file.write(v + ',' + str(label_idx))
            val_file.write('\n')

if __name__ == '__main__':
    video_root = sys.argv[1]
    out_dir = sys.argv[2]
    label_file = sys.argv[3]
    split_datasets(video_root, out_dir, label_file)
