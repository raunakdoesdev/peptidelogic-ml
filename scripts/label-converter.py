import json
import cv2
import os
import pickle
from tqdm import tqdm

orig_label = json.load(open('benv2.json'))
video_folder = 'D:\Peptide Logic\Writhing'

for video in orig_label.keys():
    if video == 'Labeler': continue
    video_path = os.path.join(video_folder, video)
    cap = cv2.VideoCapture(video_path)

    frame2read, read2time = pickle.load(open(video_path.replace('mp4', 'map'), 'rb'))
    for frame in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        if frame2read.get(frame, "SAD") == "SAD":
            frame2read[frame] = frame2read[frame - 1]

    for behavior in orig_label[video].keys():
        orig_label[video][behavior]['Label Start'] = frame2read[int(orig_label[video][behavior]['Label Start'])]
        orig_label[video][behavior]['Label End'] = frame2read[int(orig_label[video][behavior]['Label End'])]
        for i, event_range in enumerate(tqdm(orig_label[video][behavior]['event_ranges'])):
            start, stop = event_range
            orig_label[video][behavior]['event_ranges'][i] = [frame2read[int(start)], frame2read[int(stop)]]

json.dump(orig_label, open('timelabels.json', 'w'))
