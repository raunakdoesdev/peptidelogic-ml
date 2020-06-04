import logging

import mousenet as mn
import torch
import os


# Define Network Input
def df_map(df):
    # x = [mn.dist(df, 'leftpaw', 'tail'), mn.dist(df, 'rightpaw', 'tail'), mn.dist(df, 'neck', 'tail'), body_length,
    #      df['leftpaw']['likelihood'], df['rightpaw']['likelihood']]
    x = [df['hindpaw_right']['likelihood'], df['hindpaw_left']['likelihood'], df['hindheel_right']['likelihood'],
         df['hindheel_left']['likelihood'], df['frontpaw_left']['likelihood'], df['frontpaw_right']['likelihood'],
         mn.dist(df, 'tail', 'hindpaw_right'), mn.dist(df, 'tail', 'hindpaw_left')]
    return x


# Define hyperparameters
hparams = {'num_filters': (19, (1, 20)),
           'num_filters2': (8, (1, 20)),
           'filter_width': (101, (11, 101, 10)),  # must be an odd number
           'filter_width2': (31, (11, 101, 10)),  # must be an odd number
           'in_channels': 8,  # number of network inputs
           'weight': 7,  # how much "emphasis" to give to positive labels
           'loss': torch.nn.functional.binary_cross_entropy,
           'train_val_split': 1.0}


# Define Network Architecture
class MouseModel(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(params.in_channels, params.num_filters, kernel_size=params.filter_width,
                                     padding=(params.filter_width - 1) // 2)
        self.conv2 = torch.nn.Conv1d(params.num_filters, params.num_filters2, kernel_size=params.filter_width2,
                                     padding=(params.filter_width2 - 1) // 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x, _ = torch.max(x, dim=1)
        return torch.sigmoid(x)


def train_writhing_network():
    model_path = 'writhing.model'
    dlc = mn.DLCProject(config_path='/home/pl/pl-ml/Retraining-BenR-2020-05-25/config.yaml')

    torch.manual_seed(1)  # consistent behavior w/ random seed
    labeled_videos = mn.json_to_videos('/home/pl/Data', '../benv2-synced.json', mult=1)
    dlc.infer_trajectories(labeled_videos)
    dataset = mn.DLCDataset(labeled_videos, df_map, behavior='Writhe')
    runner = mn.Runner(MouseModel, hparams, dataset)
    if os.path.exists(model_path):
        return runner.get_model(torch.load(model_path)).cuda()
    else:
        model, auc = runner.train_model(max_epochs=500)
        torch.save(model.model.state_dict(), model_path)
        return model


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

model = train_writhing_network()

# Setup DLC Project
dlc = mn.DLCProject(config_path='/home/pl/pl-ml/Retraining-BenR-2020-05-25/config.yaml')

# Infer trajectories
videos = mn.folder_to_videos('/home/pl/Data/mWT SR 017 (PL 100960 DRC IV)_renamed', labeled=True)
dlc.infer_trajectories(videos)

infer_dataset = mn.DLCDataset(videos, df_map, only_x=True)
videos = infer_dataset.videos
predictions = model(infer_dataset[0].cuda()).cpu().detach().numpy()
import numpy as np
import scipy.signal

result = {}
for video_num, video in enumerate(videos):
    video: mn.LabeledVideo = video
    try:
        treatment = map[video.get_name().split('.')[0]]
        if treatment == 'SKIP':
            continue
        # predictions[video_num] = predictions[video_num] > 0.7
        # for i in range(1, len(predictions[video_num])):
        #     predictions[video_num][i] += predictions[video_num][i - 1]
        # pred = predictions[video_num]
        peaks, _ = scipy.signal.find_peaks(predictions[video_num], height=0.9)
        pred = len(peaks)
        print(f'{video.path} {len(peaks)}')
        print(peaks)
        break

        if treatment not in result:
            result[treatment] = (1, pred)
        else:
            prev_num, prev_mean = result[treatment]
            result[treatment] = prev_num + 1, ((prev_mean * prev_num) + pred) / (prev_num + 1)
    except KeyError:
        pass
# print(result)
import pickle
pickle.dump((predictions, predictions), open('vis2.pkl', 'wb'))

# import matplotlib.pyplot as plt
#
# for treatment in result.keys():
#     print(f'{treatment} -> {result[treatment]}')
#     _, imp = result[treatment]
#     plt.plot(list(range(len(imp))), imp, label=treatment)
#
# plt.legend()
# plt.savefig('test.png')
