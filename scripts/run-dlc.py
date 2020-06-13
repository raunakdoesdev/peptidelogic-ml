import logging
import pickle
import xgboost as xgb
import numpy as np
import mousenet as mn

map = {'BW_MWT_191104_M4_R1': 'Vehicle(N/A)',
       'BW_MWT_191104_M2_R2': 'Vehicle(N/A)',
       'BW_MWT_191104_M5_R2': 'Vehicle(N/A)',
       'BW_MWT_191104_M3_R3': 'Vehicle(N/A)',
       'BW_MWT_191104_M5_R3': 'SKIP',
       'BW_MWT_191105_M1_R1': 'SKIP',
       'BW_MWT_191105_M2_R1': 'SKIP',
       'BW_MWT_191105_M1_R2': 'Vehicle(N/A)',
       'BW_MWT_191105_M6_R2': 'SKIP',
       'BW_MWT_191105_M1_R3': 'Vehicle(N/A)',
       'BW_MWT_191105_M3_R3': 'Vehicle(N/A)',
       'BW_MWT_191107_M5_R1': 'SKIP',
       'BW_MWT_191107_M6_R1': 'SKIP',
       'BW_MWT_191107_M4_R2': 'Vehicle(N/A)',
       'BW_MWT_191107_M5_R2': 'Vehicle(N/A)',
       'BW_MWT_191107_M5_R3': 'Vehicle(N/A)',
       'BW_MWT_191107_M6_R3': 'Vehicle(N/A)',
       'BW_MWT_191107_M4_R1': 'SKIP',
       'BW_MWT_191107_M6_R2': 'PL 100,960(3 nmole/kg)',
       'BW_MWT_191107_M4_R3': 'PL 100,960(3 nmole/kg)',
       'BW_MWT_191104_M6_R1': 'SKIP',
       'BW_MWT_191104_M3_R1': 'PL 100,960(10 nmole/kg)',
       'BW_MWT_191104_M1_R2': 'PL 100,960(10 nmole/kg)',
       'BW_MWT_191104_M4_R2': 'SKIP',
       'BW_MWT_191104_M2_R3': 'PL 100,960(10 nmole/kg)',
       'BW_MWT_191105_M3_R1': 'SKIP',
       'BW_MWT_191105_M5_R2': 'PL 100,960(10 nmole/kg)',
       'BW_MWT_191105_M6_R3': 'PL 100,960(10 nmole/kg)',
       'BW_MWT_191107_M3_R1': 'SKIP',
       'BW_MWT_191107_M1_R2': 'SKIP',
       'BW_MWT_191104_M2_R1': 'PL 100,960(30 nmole/kg)',
       'BW_MWT_191104_M3_R2': 'PL 100,960(30 nmole/kg)',
       'BW_MWT_191104_M4_R3': 'PL 100,960(30 nmole/kg)',
       'BW_MWT_191105_M5_R1': 'SKIP',
       'BW_MWT_191105_M6_R1': 'SKIP',
       'BW_MWT_191105_M2_R2': 'PL 100,960(30 nmole/kg)',
       'BW_MWT_191105_M4_R2': 'PL 100,960(30 nmole/kg)',
       'BW_MWT_191105_M5_R3': 'PL 100,960(30 nmole/kg)',
       'BW_MWT_191107_M2_R1': 'SKIP',
       'BW_MWT_191104_M1_R1': 'PL 100,960(100 nmole/kg)',
       'BW_MWT_191104_M5_R1': 'PL 100,960(100 nmole/kg)',
       'BW_MWT_191104_M6_R2': 'PL 100,960(100 nmole/kg)',
       'BW_MWT_191104_M1_R3': 'PL 100,960(100 nmole/kg)',
       'BW_MWT_191104_M6_R3': 'PL 100,960(100 nmole/kg)',
       'BW_MWT_191105_M4_R1': 'SKIP',
       'BW_MWT_191105_M3_R2': 'PL 100,960(100 nmole/kg)',
       'BW_MWT_191105_M2_R3': 'PL 100,960(100 nmole/kg)',
       'BW_MWT_191105_M4_R3': 'PL 100,960(100 nmole/kg)',
       'BW_MWT_191107_M1_R1': 'SKIP',
       'BW_MWT_191107_M2_R2': 'PL 100,960(300 nmole/kg)',
       'BW_MWT_191107_M3_R2': 'PL 100,960(300 nmole/kg)',
       'BW_MWT_191107_M1_R3': 'PL 100,960(300 nmole/kg)',
       'BW_MWT_191107_M2_R3': 'PL 100,960(300 nmole/kg)',
       'BW_MWT_191107_M3_R3': 'PL 100,960(300 nmole/kg)', }

# Setup DLC Project
dlc = mn.DLCProject(config_path='/home/pl/Retraining-BenR-2020-05-25/config.yaml')
# Infer trajectories
videos = mn.folder_to_videos('/home/pl/Data/mWT SR 017 (PL 100960 DRC IV)_renamed', skip_words=('labeled', 'CFS'),
                             labeled=True)
dlc.infer_trajectories(videos, infer=True)

# Load Model
clf = pickle.load(open('XGB.save', 'rb'))

from tqdm import tqdm
result = {}
for video_num, video in enumerate(tqdm(videos)):

    video: mn.LabeledVideo = video
    try:
        treatment = map[video.get_name().split('.')[0]]
        y_pred = clf.predict(xgb.DMatrix(data=mn.video_to_dataset(video, window_size=100)))[:50000]
        y_pred_cumulative = np.cumsum(y_pred)

        if treatment == 'SKIP':
            continue

        if treatment not in result:
            result[treatment] = (1, y_pred_cumulative)
        else:
            prev_num, prev_mean = result[treatment]
            result[treatment] = prev_num + 1, ((prev_mean * prev_num) + y_pred_cumulative) / (prev_num + 1)
    except Exception as e:
        # logging.exception(e)
        pass
import matplotlib.pyplot as plt

for treatment in result.keys():
    _, y_pred_cumulative = result[treatment]
    plt.plot(list(range(len(y_pred_cumulative))), y_pred_cumulative, label=treatment)

plt.legend()
plt.savefig('/home/pl/Desktop/test.png')
