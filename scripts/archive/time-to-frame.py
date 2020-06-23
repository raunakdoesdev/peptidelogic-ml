import json
import cv2
import os
import pickle
from tqdm import tqdm

orig_label = json.load(open('../2020-06-03_ben.json'))
video_folder = '/home/pl/Data/mWT SR 017 (PL 100960 DRC IV)_renamed'

for video in orig_label.keys():
    if video == 'Labeler': continue
    video_path = os.path.join(video_folder, video)
    cap = cv2.VideoCapture(video_path)


    def convert(frame):
        cap.set(cv2.CAP_PROP_POS_MSEC, frame)
        return cap.get(cv2.CAP_PROP_POS_FRAMES)


    for behavior in orig_label[video].keys():
        orig_label[video][behavior]['Label Start'] = convert(orig_label[video][behavior]['Label Start'])
        orig_label[video][behavior]['Label End'] = convert(orig_label[video][behavior]['Label End'])
        for i, event_range in enumerate(tqdm(orig_label[video][behavior]['event_ranges'])):
            start, stop = event_range
            orig_label[video][behavior]['event_ranges'][i] = [convert(start), convert(stop)]

json.dump(orig_label, open('linux-frame_nums.json', 'w'))
