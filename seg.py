#encoding=utf8
import cv2
import os
import sys
import time
import numpy as np

def get_action(frameCnt, annos):
    for anno in annos:
        if frameCnt <= anno['end'] and frameCnt >= anno['start']:
            return anno['action']
    return 'normal'

def save_mp4(seg_frames, out_dir, action, video_name_no_ext):
    if action == 'normal' and len(seg_frames) < 20:
        return

    if action == 'normal' and len(seg_frames) > 500:
        idx = range(len(seg_frames))
        selected_idx = np.random.choice(idx, 500)
        selected_idx = sorted(selected_idx)
        seg_frames = [seg_frames[i] for i in selected_idx]

    ts = int(round(time.time() * 1000))
    seg_dir = os.path.join(out_dir, action)
    if not os.path.isdir(seg_dir):
        os.makedirs(seg_dir)
    seg_file = os.path.join(seg_dir, '%s_%d.mp4' % (video_name_no_ext,ts))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(seg_file, fourcc, 25.0, (seg_frames[0].shape[1], seg_frames[0].shape[0]))
    for frame in seg_frames:
        out.write(frame)
    out.release()

def main(video_root, anno_csv, out_dir):
    video_paths = [ os.path.join(video_root, x) for x in os.listdir(video_root) if '.mp4' in x or '.avi' in x]
    annos = {}
    with open(anno_csv) as f:
        for l in f:
            l = l.strip()
            if '动作' in l:
                continue
            if l:
                fields = [ x for x in l.split(',') if x ]
                if fields[0] not in annos:
                    annos[fields[0]] = []
                start = int(fields[3])
                end = int(fields[4])
                if start > end:
                    start,end = end, start
                annos[fields[0]].append({'start': start, 'end': end, 'action': fields[5]})


    for idx, video_path in enumerate(video_paths):
        video = cv2.VideoCapture(video_path)
        totalFrameNum = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(video.get(cv2.CAP_PROP_FPS))

        video_name = video_path.split('/')[-1]
        video_name_no_ext = video_name.split('.')[0]
        anno = annos[video_name]

        frames = []
        action_clip = []
        normal_clip = []
        for frameCnt in range(totalFrameNum):
            ok, frame = video.read()
            if not ok:
                continue

            action = get_action(frameCnt, anno)

            if action == 'normal':
                normal_clip.append(frame)
                if action_clip:
                    pre_action = get_action(frameCnt - 1, anno)
                    save_mp4(action_clip, out_dir, pre_action, video_name_no_ext)
                    action_clip = []
            else:
                action_clip.append(frame)
                if normal_clip:
                    pre_action = get_action(frameCnt - 1, anno)
                    save_mp4(normal_clip, out_dir, pre_action, video_name_no_ext)
                    normal_clip = []

        if normal_clip:
            save_mp4(normal_clip, out_dir, 'normal', video_name_no_ext)
        if action_clip:
            save_mp4(action_clip, out_dir, action, video_name_no_ext)

        print('finish %s' % video_path)

if __name__ == '__main__':
    video_root = sys.argv[1]
    anno_csv = sys.argv[2]
    out_dir = sys.argv[3]
    main(video_root, anno_csv, out_dir)
