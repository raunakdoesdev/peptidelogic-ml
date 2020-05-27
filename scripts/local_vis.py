import mousenet as mn
import torch
import pandas as pd
import logging
from tqdm import tqdm
import cv2


import pickle

y, y_hat = pickle.load(open('vis.pkl', 'rb'))

labeled_videos = mn.json_to_videos(r"D:\Peptide Logic\Writhing", '../benv2-synced.json')

labeled_videos[0].df_path = r'D:\Peptide Logic\Writhing\BW_MWT_191107_M4_R2DeepCut_resnet50_mouse_behavior_idJan24shuffle1_200000.h5'
labeled_videos[1].df_path = r'D:\Peptide Logic\Writhing\BW_MWT_191107_M5_R2DeepCut_resnet50_mouse_behavior_idJan24shuffle1_200000.h5'
labeled_videos[0].calculate_mappings()
labeled_videos[1].calculate_mappings()

i = 1
video = labeled_videos[i]
mn.VisualDebugger(video, y[i], y_hat[i])
