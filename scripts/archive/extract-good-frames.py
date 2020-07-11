import logging
import os

import torch

import mousenet as mn

logging.getLogger().setLevel(logging.DEBUG)  # Log all info

# Setup DLC Project
dlc = mn.DLCProject(config_path='/home/pl/sauhaarda/peptide_logic_refactored/dlc/'
                                'mouse_behavior_id-sauhaarda-2020-01-24/config.yaml', pcutoff=0.25)

labeled_videos = mn.json_to_videos('/home/pl/Data', '../benv2-synced.json', mult=1)
dlc.infer_trajectories(labeled_videos)

behavior = 'Itch'

# Videos without human label
# videos = mn.folder_to_videos('/home/pl/Data/mWT SR 017 (PL 100960 DRC IV)', labeled=True, required_words=['199107'])
videos = mn.folder_to_videos('/home/pl/Data/200317', labeled=True)

dlc.infer_trajectories(videos)
for video in videos:
    video.calculate_mappings()

# Infer trajectories
# Define hyperparameters
writhing_hparams = {'num_filters': (7, (1, 20)),
                    'num_filters2': (11, (1, 20)),
                    'filter_width': (71, (11, 101, 10)),  # must be an odd number
                    'filter_width2': (91, (11, 101, 10)),  # must be an odd number
                    'in_channels': 6,  # number of network inputs
                    'weight': 7,  # how much "emphasis" to give to positive labels
                    'loss': torch.nn.functional.binary_cross_entropy,
                    'train_val_split': 1.0}

itching_hparams = {'num_filters': (15, (1, 20)),
                   'num_filters2': (7, (1, 20)),
                   'filter_width': (21, (11, 101, 10)),  # must be an odd number
                   'filter_width2': (61, (11, 101, 10)),  # must be an odd number
                   'in_channels': 6,  # number of network inputs
                   'weight': 7,  # how much "emphasis" to give to positive labels
                   'loss': torch.nn.functional.binary_cross_entropy,
                   'train_val_split': 1.0}

if behavior == 'Itch':
    hparams = itching_hparams
elif behavior == 'Writhe':
    hparams = writhing_hparams


# Define Network Input
def df_map(df):
    df['head', 'x'] = (df['leftear']['x'] + df['rightear']['x']) / 2
    df['head', 'y'] = (df['leftear']['y'] + df['rightear']['y']) / 2
    body_length = mn.dist(df, 'head', 'tail')
    x = [mn.dist(df, 'leftpaw', 'tail'), mn.dist(df, 'rightpaw', 'tail'), mn.dist(df, 'neck', 'tail'), body_length,
         df['leftpaw']['likelihood'], df['rightpaw']['likelihood']]
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

model_path = f'{behavior}.weights'
dataset = mn.DLCDataset(labeled_videos, df_map, behavior=behavior)
runner = mn.Runner(MouseModel, hparams, dataset)
if not os.path.exists(model_path):
    model, auc = runner.train_model(max_epochs=500)
    runner = mn.Runner(MouseModel, hparams, dataset)
    torch.save(model.model.state_dict(), model_path)
else:
    model = runner.get_model(torch.load(model_path)).cuda()

# Infer on All Videos
infer_dataset = mn.DLCDataset(videos, df_map, only_x=True)
predictions = model(infer_dataset[0].cuda()).cpu().detach().numpy()

import cv2
import numpy as np
import shutil

# if os.path.exists('save_frames'):
#     shutil.rmtree('save_frames')
# os.makedirs('save_frames', exist_ok=True)

videos = infer_dataset.videos


def check_range(added_frames, proposal, threshold=20):
    for frame in added_frames:
        if abs(proposal - frame) < threshold:
            return False
    return True


for video_num, video in enumerate(videos):

    added_frames = []

    video_name = os.path.basename(video.path).split('.mp4')[0]
    num_frames = 50
    idxes = np.argpartition(predictions[video_num], -num_frames)[-num_frames:]
    last = float("inf")

    for idx in idxes:
        if predictions[video_num][idx] > 0.9 and check_range(added_frames, idx):
            added_frames.append(idx)
            cap = cv2.VideoCapture(video.path)
            last = idx
            try:
                cap.set(cv2.CAP_PROP_POS_MSEC, video.read2time[idx])
                _, image = cap.read()
                cv2.imwrite(f'save_frames/{video_name}frame{idx}.png', image)
                print(
                    f'Frame #{idx} in Video {videos[video_num].path} is likely {behavior}! Probability = {predictions[video_num][idx]}')
            except KeyError:
                pass
