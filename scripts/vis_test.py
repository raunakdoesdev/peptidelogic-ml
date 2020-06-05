import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.absolute().parent.absolute()))

import logging

import torch

import mousenet as mn

video_num = 0  # <-- WHICH VIDEO YOU WANT TO VISUALIZE (0, 1, 2, etc.)
SCALING = 0.6

logging.getLogger().setLevel(logging.DEBUG)  # Log all info

dlc = mn.DLCProject(config_path='/home/pl/Retraining-BenR-2020-05-25/config.yaml')
labeled_videos = mn.folder_to_videos('/home/pl/Data/mWT SR 017 (PL 100960 DRC IV)_renamed', labeled=True)

# Infer trajectories
dlc.infer_trajectories(labeled_videos)

torch.manual_seed(1)  # consistent behavior w/ random seed

video = labeled_videos[video_num]
video.calculate_mappings()
video: mn.LabeledVideo = video
import pickle
print(video.ground_truth)
video.start = 0
video.end = 10000
# y = pickle.load(video.ground_truth['Writhe'])
y = [1] * 10000
mn.VisualDebugger(video, y, y, scaling=SCALING)
