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
    def __init__(self, videos, input_map, behavior='Writhe', multiplier=1.0):
        self.x = []
        self.y = []
        self.videos = videos
        self.mx = None
        self.my = None

        if behavior is not None:
            self.video_splits = []
            for video in videos:
                ground_truth = torch.load(video.ground_truth[behavior])
                df = pd.read_hdf(video.df_path)
                df = df[df.columns.get_level_values(0).unique()[0]]
                df = df.iloc[int(video.start): int(video.end)]
                self.x.append(torch.cat(
                    [F.normalize(torch.FloatTensor(flag.to_numpy()), dim=0).unsqueeze(0) for flag in input_map(df)]))
                self.y.append(ground_truth)

    def __len__(self):
        return 1

    def _merge(self):
        self.mx = pad_sequence([x.permute(1, 0) for x in self.x], batch_first=True).permute(0, 2, 1)
        self.my = pad_sequence(self.y, batch_first=True)

    def __getitem__(self, idx):
        if self.mx is None:
            self._merge()

        return self.mx, self.my

    # noinspection PyTypeChecker
    def split_dataset(self, train_val_split):
        x1, x2, y1, y2 = [], [], [], []
        for i in range(len(self.x)):
            size = round((1.0 - train_val_split) * self.y[i].shape[0])
            x1.append(self.x[i][:, :size])
            x2.append(self.x[i][:, size:])
            y1.append(self.y[i][:size])
            y2.append(self.y[i][size:])

        d1 = DLCDataset(None, None, behavior=None)
        d2 = DLCDataset(None, None, behavior=None)
        d1.x, d2.x = x1, x2
        d1.y, d2.y = y1, y2

        return d2, d1
