import logging
import mousenet as mn
import torch

logging.getLogger().setLevel(logging.DEBUG)  # Log all info

labeled_videos = mn.json_to_videos('videos/', 'benv2.json', mult=30/29.884408054387492)

labeled_videos[
    0].labeled_video_path = 'videos/BW_MWT_191107_M4_R2.mp4'

import pickle

model_outputs = pickle.load(open('videos/temp2.pkl', 'rb'))

for video, y, y_hat in zip(labeled_videos, *model_outputs):
    mn.VisualDebugger(video, y, y_hat, div=30/29.884408054387492)
