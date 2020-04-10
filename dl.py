import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import os
from torch.nn import functional as F
from torch.utils.data import random_split
import torchvision.transforms as transforms
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm

def process_data(dataset_path, n, dlc_flags):
    dataset = pd.read_pickle(dataset_path)

    print("Finding Video Boundaries")
    
    idx, cur_val = 0, 0
    startstops = []
    video_names = []

    with tqdm(total=len(dataset) - 1) as pbar:
        while idx < len(dataset):
            pbar.update(1)
            if dataset.iloc[idx]['video_name'] != dataset.iloc[idx + 1]['video_name']:
                video_names.append(dataset.iloc[idx]['video_name'])
                startstops.append((cur_val, idx + 1))
                cur_val = idx + 1
                idx += int(0.9 * startstops[0][1])
                pbar.update(int(0.9 * startstops[0][1]))
            idx += 1
            pbar.update(1)

    startstops.append((cur_val, len(dataset)))

    print(video_names)

    for start, stop in startstops:
        tensors = []
        for dlc_flag in dlc_flags:
            tensors.append(torch.FloatTensor(dataset.iloc[start : stop][dlc_flag].values).unsqueeze(0))
        data.append((torch.cat(tensors, 0).unsqueeze(0), torch.FloatTensor(dataset.iloc[start : stop]['machine_label'].values).unsqueeze(0)))

    pickle.dump(data, open('processed_data_new.pkl', 'wb'))

class ItchingDataset(Dataset):
    """Mouse Itching Datset."""


    def __init__(self, dataset_path, n, dlc_flags, process=False):
        """
        dataset_path : str - path to dataset pickle file
        n : int - frame radius (how many frames to look at to the left/right)
        """

        self.data = []
        self.n = n
        self.dlc_flags = dlc_flags

        if process:
            self.process_data(dataset_path, n, dlc_flags, process)
        else:
            self.data = pickle.load(open(dataset_path, 'rb'))
        

  
    def __len__(self):
        return len(self.data)  

    def __getitem__(self, idx):
        return self.data[idx]


class ItchDetector(pl.LightningModule):
    def __init__(self, dataset_path, n, dlc_flags, num_train=10, num_val=4):
        """
        dataset_path : str - path to dataset pickle file
        n : int - frame radius (how many frames to look at to the left/right)
        dlc_flags | iterable - iterable with strings for each dlc flag
        """
        super().__init__()

        self.dataset_path = dataset_path
        self.n = n
        self.dlc_flags = dlc_flags
        self.num_train = num_train
        self.num_val = num_val
        self.trainset = None
        self.valset = None


        # Define network architecture
        self.l1 = torch.nn.Conv2d(1, 1, kernel_size=(len(dlc_flags), (2 * n + 1)), padding=(0, n), bias=False)

    def get_dataset(self):
        self.main_dataset = ItchingDataset(self.dataset_path, self.n, self.dlc_flags)
        self.trainset, self.valset, _ = random_split(self.main_dataset, [self.num_train, self.num_val, len(self.main_dataset) - self.num_train - self.num_val])

    def forward(self, x):
        out =  torch.relu(self.l1(x.view(1, 1, len(self.dlc_flags), -1))).mean(1).view(1, 1, -1)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        pred_y = y_hat >= 0.5
        return {'loss': F.mse_loss(y_hat, y), 'accuracy' : torch.sum(y == pred_y).item() / y.numel()}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        pred_y = y_hat >= 0.5
        return {'loss': F.mse_loss(y_hat, y), 'accuracy' : torch.sum(y == pred_y).item() / y.numel()}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = np.mean([x['accuracy'] for x in outputs])
        tensorboard_logs = {'val_loss' : avg_loss, 'accuracy' : avg_acc}
        return {'avg_val_loss' : avg_loss, 'log' : tensorboard_logs}

    def train_dataloader(self):
        if self.trainset is None:
            self.get_dataset()
        return DataLoader(self.trainset, batch_size=1)

    def val_dataloader(self):
        if self.valset is None:
            self.get_dataset()
        return DataLoader(self.valset, batch_size=1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


if __name__ == "__main__":
    dlc_flags = ('leftpaw_prob', 'rightpaw_prob')
    model = ItchDetector('processed_data.pkl', 5, dlc_flags)
    trainer = pl.Trainer(gpus=1, benchmark=True, max_epochs=150)
    trainer.fit(model)

    # for filter_num in range(len(model.l1.weight[0])):
    #     for flag_i, dlc_flag in enumerate(dlc_flags):
    #         plt.plot(model.l1.weight[0][filter_num][flag_i].data.cpu(), label=f"F{filter_num} {dlc_flag}")

    # plt.title("Filter Visualization")
    # plt.xlabel("Frame")
    # plt.ylabel("Weight")
    # plt.legend()
    # plt.savefig('filter.png')

    vids = ['BW_MIT_200304_M1_R1', 'BW_MIT_200304_M1_R2', 'BW_MIT_200304_M1_R3', 'BW_MIT_200304_M1_R4', 'BW_MIT_200304_M2_R1', 'BW_MIT_200304_M2_R2', 'BW_MIT_200304_M2_R3', 'BW_MIT_200304_M2_R4', 'BW_MIT_200304_M3_R1', 'BW_MIT_200304_M3_R2', 'BW_MIT_200304_M3_R3', 'BW_MIT_200304_M3_R4', 'BW_MIT_200304_M4_R1', 'BW_MIT_200304_M4_R2', 'BW_MIT_200304_M4_R3', 'BW_MIT_200304_M4_R4', 'BW_MIT_200304_M5_R1', 'BW_MIT_200304_M5_R2', 'BW_MIT_200304_M5_R3', 'BW_MIT_200304_M5_R4', 'BW_MIT_200304_M6_R1', 'BW_MIT_200304_M6_R2', 'BW_MIT_200304_M6_R3', 'BW_MIT_200304_M6_R4', 'BW_MIT_200305_M1_R1', 'BW_MIT_200305_M1_R2', 'BW_MIT_200305_M1_R3', 'BW_MIT_200305_M1_R4', 'BW_MIT_200305_M2_R1', 'BW_MIT_200305_M2_R2', 'BW_MIT_200305_M2_R3', 'BW_MIT_200305_M2_R4', 'BW_MIT_200305_M3_R1', 'BW_MIT_200305_M3_R2', 'BW_MIT_200305_M3_R3', 'BW_MIT_200305_M3_R4', 'BW_MIT_200305_M4_R1', 'BW_MIT_200305_M4_R2', 'BW_MIT_200305_M4_R3', 'BW_MIT_200305_M4_R4', 'BW_MIT_200305_M5_R1', 'BW_MIT_200305_M5_R2', 'BW_MIT_200305_M5_R3', 'BW_MIT_200305_M5_R4', 'BW_MIT_200305_M6_R1', 'BW_MIT_200305_M6_R2', 'BW_MIT_200305_M6_R3', 'BW_MIT_200305_M6_R4', 'BW_MIT_200306_M1_R1', 'BW_MIT_200306_M1_R2', 'BW_MIT_200306_M1_R3', 'BW_MIT_200306_M1_R4', 'BW_MIT_200306_M2_R1', 'BW_MIT_200306_M2_R2', 'BW_MIT_200306_M2_R3', 'BW_MIT_200306_M2_R4', 'BW_MIT_200306_M3_R1', 'BW_MIT_200306_M3_R2', 'BW_MIT_200306_M3_R3', 'BW_MIT_200306_M3_R4', 'BW_MIT_200306_M4_R1', 'BW_MIT_200306_M4_R2', 'BW_MIT_200306_M4_R3', 'BW_MIT_200306_M4_R4', 'BW_MIT_200306_M5_R1', 'BW_MIT_200306_M5_R2', 'BW_MIT_200306_M5_R3', 'BW_MIT_200306_M5_R4', 'BW_MIT_200306_M6_R1', 'BW_MIT_200306_M6_R2', 'BW_MIT_200306_M6_R3', 'BW_MIT_200306_M6_R4', 'BW_MIT_200317_M1_R_1', 'BW_MIT_200317_M1_R_2', 'BW_MIT_200317_M1_R_3', 'BW_MIT_200317_M1_R_4', 'BW_MIT_200317_M2_R_1', 'BW_MIT_200317_M2_R_2', 'BW_MIT_200317_M2_R_3', 'BW_MIT_200317_M2_R_4', 'BW_MIT_200317_M3_R_1', 'BW_MIT_200317_M3_R_2', 'BW_MIT_200317_M3_R_3', 'BW_MIT_200317_M3_R_4', 'BW_MIT_200317_M4_R_1', 'BW_MIT_200317_M4_R_2', 'BW_MIT_200317_M4_R_3', 'BW_MIT_200317_M4_R_4', 'BW_MIT_200317_M5_R_1', 'BW_MIT_200317_M5_R_2', 'BW_MIT_200317_M5_R_3', 'BW_MIT_200317_M5_R_4', 'BW_MIT_200317_M6_R_1', 'BW_MIT_200317_M6_R_2', 'BW_MIT_200317_M6_R_3', 'BW_MIT_200317_M6_R_4', 'BW_MIT_200318_M1_R_1', 'BW_MIT_200318_M1_R_2', 'BW_MIT_200318_M1_R_3', 'BW_MIT_200318_M1_R_4', 'BW_MIT_200318_M2_R_1', 'BW_MIT_200318_M2_R_2', 'BW_MIT_200318_M2_R_3', 'BW_MIT_200318_M2_R_4', 'BW_MIT_200318_M3_R_1', 'BW_MIT_200318_M3_R_2', 'BW_MIT_200318_M3_R_3', 'BW_MIT_200318_M3_R_4', 'BW_MIT_200318_M4_R_1', 'BW_MIT_200318_M4_R_2', 'BW_MIT_200318_M4_R_3', 'BW_MIT_200318_M4_R_4', 'BW_MIT_200318_M5_R_1', 'BW_MIT_200318_M5_R_2', 'BW_MIT_200318_M5_R_3', 'BW_MIT_200318_M5_R_4', 'BW_MIT_200318_M6_R_1', 'BW_MIT_200318_M6_R_2', 'BW_MIT_200318_M6_R_3', 'BW_MIT_200318_M6_R_4', 'BW_MIT_200319_M1_R_1', 'BW_MIT_200319_M1_R_2', 'BW_MIT_200319_M1_R_3', 'BW_MIT_200319_M1_R_4', 'BW_MIT_200319_M2_R_1', 'BW_MIT_200319_M2_R_2', 'BW_MIT_200319_M2_R_3', 'BW_MIT_200319_M2_R_4', 'BW_MIT_200319_M3_R_1', 'BW_MIT_200319_M3_R_2', 'BW_MIT_200319_M3_R_3', 'BW_MIT_200319_M3_R_4', 'BW_MIT_200319_M4_R_1', 'BW_MIT_200319_M4_R_2', 'BW_MIT_200319_M4_R_3', 'BW_MIT_200319_M4_R_4', 'BW_MIT_200319_M5_R_1', 'BW_MIT_200319_M5_R_2', 'BW_MIT_200319_M5_R_3', 'BW_MIT_200319_M5_R_4', 'BW_MIT_200319_M6_R_1', 'BW_MIT_200319_M6_R_2', 'BW_MIT_200319_M6_R_3']
    
    idx = vids.index('BW_MIT_200318_M6_R_3')

    with open('predic.txt', 'w') as f:
        out = (model(model.main_dataset[idx][0].unsqueeze(0).cuda()) > 0.5)[0].cpu().numpy().tolist()
        for i, val in enumerate(out[0]):
            if val:
                f.write(f'{i}\n') 
