from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from mousenet import util
import logging
import pytorch_lightning as pl
from torch.nn import functional as F


class DLCDataset(Dataset):
    # noinspection PyArgumentList
    def __init__(self, videos, input_map, behavior='Writhe', multiplier=1.0):
        if behavior is not None:
            self.x = []
            self.y = []

            for video in videos:
                ground_truth = torch.load(video.ground_truth[behavior])
                df = pd.read_hdf(video.df_path)

                df = df[df.columns.get_level_values(0).unique()[0]]
                df = df.iloc[int(video.start * multiplier): int(video.end * multiplier)]
                self.x.append(torch.cat([torch.FloatTensor(flag.to_numpy()).unsqueeze(0) for flag in input_map(df)]))
                self.y.append(ground_truth)

            # add a segment of zeros between elements
            self.x = util.intersperse(self.x, torch.zeros([self.x[0].shape[0], 100]))
            self.y = util.intersperse(self.y, torch.zeros([100]))

            # concatenate across videos
            self.x = torch.cat(self.x, dim=1)
            self.y = torch.cat(self.y)

            logging.debug(f'Input shape is {self.x.shape}')
            logging.debug(f'output shape is {self.y.shape}')

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.x, self.y

    # noinspection PyTypeChecker
    def split_dataset(self, train_val_split):
        size1 = round(train_val_split * self.y.shape[0])

        d1 = DLCDataset(None, None, behavior=None)
        d2 = DLCDataset(None, None, behavior=None)
        d1.x, d2.x = self.x[:, :size1], self.x[:, size1:]
        d1.y, d2.y = self.y[:size1], self.y[size1:]
        return d1, d2


class ItchDetector(pl.LightningModule):
    def __init__(self, dataset, loss_func=F.binary_cross_entropy, train_val_split=0.7):
        super().__init__()
        self.loss_func = loss_func
        self.train_set, self.val_set = dataset.split_dataset(train_val_split)

        num_filters = 4
        filter_width = 1
        in_channels = dataset[0][0].shape[0]
        self.model = torch.nn.Conv1d(in_channels, num_filters, kernel_size=2 * filter_width + 1,
                            padding=filter_width)

    def forward(self, x):
        x = self.model(x)
        x = x.max(dim=1)
        return F.sigmoid(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'loss': self.loss_func(y_hat, y)}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'loss': self.loss_func(y_hat, y)}

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=1)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
