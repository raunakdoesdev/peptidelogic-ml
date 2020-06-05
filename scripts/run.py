import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.absolute().parent.absolute()))

import logging

import torch

import mousenet as mn

video_num = 0  # <-- WHICH VIDEO YOU WANT TO VISUALIZE (0, 1, 2, etc.)
SCALING = 0.8

logging.getLogger().setLevel(logging.DEBUG)  # Log all info

dlc = mn.DLCProject(config_path='/home/pl/Retraining-BenR-2020-05-25/config.yaml')
labeled_videos = mn.json_to_videos('/home/pl/Data', '../benv2-synced.json', mult=1)

# Infer trajectories
dlc.infer_trajectories(labeled_videos)
print(labeled_videos[0].df_path)
print(labeled_videos[1].df_path)

# Define hyperparameters
writhing_hparams = {'num_filters': (19, (1, 20)),
                    'num_filters2': (8, (1, 20)),
                    'filter_width': (101, (11, 101, 10)),  # must be an odd number
                    'filter_width2': (31, (11, 101, 10)),  # must be an odd number
                    'in_channels': 8,  # number of network inputs
                    'weight': 7,  # how much "emphasis" to give to positive labels
                    'loss': torch.nn.functional.binary_cross_entropy,
                    'percent_data': None}

itching_hparams = {'num_filters': (15, (1, 20)),
                   'num_filters2': (7, (1, 20)),
                   'filter_width': (21, (11, 101, 10)),  # must be an odd number
                   'filter_width2': (61, (11, 101, 10)),  # must be an odd number
                   'in_channels': 8,  # number of network inputs
                   'weight': 7,  # how much "emphasis" to give to positive labels
                   'loss': torch.nn.functional.binary_cross_entropy,
                   'train_val_split': 1.0}

behvaior = 'Writhe'
hparams = writhing_hparams if behvaior == 'Writhe' else itching_hparams


# Define Network Input
def df_map(df):
    # x = [mn.dist(df, 'leftpaw', 'tail'), mn.dist(df, 'rightpaw', 'tail'), mn.dist(df, 'neck', 'tail'), body_length,
    #      df['leftpaw']['likelihood'], df['rightpaw']['likelihood']]
    print(df.head(2))
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

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x, _ = torch.max(x, dim=1)
        return torch.sigmoid(x)


torch.manual_seed(1)  # consistent behavior w/ random seed
dataset = mn.DLCDataset(labeled_videos, df_map, behavior='Writhe')

runner = mn.Runner(MouseModel, hparams, dataset)
model, auc = runner.train_model(max_epochs=500)
print(auc)

model_out = model(dataset[0][0].cuda()).cpu().detach().numpy()  # get model output
y, y_hat = dataset[0][1].cpu().detach().numpy(), model_out

video = labeled_videos[video_num]
video.calculate_mappings()
mn.VisualDebugger(video, y[video_num], y_hat[video_num], scaling=SCALING)
