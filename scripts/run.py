import logging

import torch

import mousenet as mn
import warnings

warnings.filterwarnings('always')

logging.getLogger().setLevel(logging.DEBUG)  # Log all info

import os

if os.name == 'nt':
    dlc = mn.DLCProject(config_path='D:\Peptide Logic\Writhing\config.yaml', pcutoff=0.25)

    labeled_videos = mn.json_to_videos(r'D:\Peptide Logic\Writhing', '../benv2-synced.json', mult=1)

else:
    dlc = mn.DLCProject(config_path='/home/pl/pl-ml/Retraining-BenR-2020-05-25/config.yaml')
    # dlc = mn.DLCProject(config_path='/home/pl/sauhaarda/peptide_logic_refactored/dlc/'
    #                                 'mouse_behavior_id-sauhaarda-2020-01-24/config.yaml', pcutoff=0.25)
    labeled_videos = mn.json_to_videos('/home/pl/Data', '../benv2-synced.json', mult=1)

# Infer trajectories
dlc.infer_trajectories(labeled_videos)

# Define hyperparameters
hparams = {'num_filters': (19, (1, 20)),
           'num_filters2': (8, (1, 20)),
           'filter_width': (101, (11, 101, 10)),  # must be an odd number
           'filter_width2': (31, (11, 101, 10)),  # must be an odd number
           'in_channels': 8,  # number of network inputs
           'weight': torch.FloatTensor([1, 7, 7]).cuda(),  # how much "emphasis" to give to positive labels
           'loss': torch.nn.functional.cross_entropy,
           'train_val_split': 1.0}


# Define Network Input
def df_map(df):
    # x = [mn.dist(df, 'leftpaw', 'tail'), mn.dist(df, 'rightpaw', 'tail'), mn.dist(df, 'neck', 'tail'), body_length,
    #      df['leftpaw']['likelihood'], df['rightpaw']['likelihood']]
    x = [df['hindpaw_right']['likelihood'], df['hindpaw_left']['likelihood'], df['hindheel_right']['likelihood'],
         df['hindheel_left']['likelihood'], df['frontpaw_left']['likelihood'], df['frontpaw_right']['likelihood'],
         mn.dist(df, 'tail', 'hindpaw_right'), mn.dist(df, 'tail', 'hindpaw_left')]
    return x


# Define Network Architecture
class MouseModel(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(params.in_channels, params.num_filters, kernel_size=params.filter_width,
                                     padding=(params.filter_width - 1) // 2)
        self.conv2 = torch.nn.Conv1d(params.num_filters, params.num_filters2, kernel_size=params.filter_width2,
                                     padding=(params.filter_width2 - 1) // 2)
        self.conv3 = torch.nn.Conv1d(params.num_filters2, 3, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


torch.manual_seed(1)  # consistent behavior w/ random seed
dataset = mn.MulticlassDataset(labeled_videos, df_map)

runner = mn.Runner(MouseModel, hparams, dataset)
runner.hyperparemeter_optimization(timeout=600)

# print(dataset[0][0].shape)
model, auc = runner.train_model(max_epochs=500)
print(auc)
