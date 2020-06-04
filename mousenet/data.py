import logging

import pandas as pd
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from mousenet import util

import matplotlib.pyplot as plt


class DLCDataset(Dataset):
    # noinspection PyArgumentList
    def __init__(self, videos, input_map, behavior='Writhe', multiplier=1.0, only_x=False):
        self.x = []
        self.y = []
        self.videos = videos
        self.mx = None
        self.my = None
        self.only_x = only_x
        real_videos = []

        if behavior is not None:
            self.video_splits = []
            for video in self.videos:
                if behavior in video.ground_truth or only_x:
                    try:
                        df = pd.read_hdf(video.df_path)
                    except ValueError:
                        logging.error(f"SKIPPING VIDEO {video.path} due to corrupted data frame!")
                        continue

                    df = df[df.columns.get_level_values(0).unique()[0]]

                    if not only_x:
                        ground_truth = torch.load(video.ground_truth[behavior])
                        self.y.append(ground_truth)
                        df = df.iloc[int(video.start): int(video.end)]

                    self.x.append(torch.cat(
                        [F.normalize(torch.FloatTensor(flag.to_numpy()), dim=0).unsqueeze(0) for flag in
                         input_map(df)]))
                    real_videos.append(video)

            self.videos = real_videos

    def __len__(self):
        return 1

    def _merge(self):
        self.mx = pad_sequence([x.permute(1, 0) for x in self.x], batch_first=True).permute(0, 2, 1)
        if not self.only_x:
            self.my = pad_sequence(self.y, batch_first=True)

    def __getitem__(self, idx):
        if self.mx is None:
            self._merge()
        if self.only_x:
            return self.mx
        return self.mx, self.my

    # noinspection PyTypeChecker
    def split_dataset(self, train_val_split):
        datasets = []

        start_pos = [0] * len(self.y)
        for split in train_val_split:
            x, y = [], []
            for i in range(len(self.y)):
                size = round(split * self.y[i].shape[0])
                x.append(self.x[i][:, start_pos[i]:start_pos[i] + size])
                y.append(self.y[i][start_pos[i]:start_pos[i] + size])
                start_pos[i] = start_pos[i] + size
            split_dataset = DLCDataset(None, None, behavior=None)
            split_dataset.x = x
            split_dataset.y = y
            datasets.append(split_dataset)

        return datasets
