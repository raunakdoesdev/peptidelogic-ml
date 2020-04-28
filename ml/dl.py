import pickle

import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torchvision import transforms


class ItchingDataset(Dataset):
    """Mouse Itching Datset."""

    def __init__(self, dataset_path, n, dlc_flags):
        """
        dataset_path : str - path to dataset pickle file
        n : int - frame radius (how many frames to look at to the left/right)
        """

        self.data = []
        self.n = n
        self.dlc_flags = dlc_flags
        self.data = pickle.load(open(dataset_path, 'rb'))[:1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        x = x.view(1, len(self.dlc_flags), -1)
        x = F.normalize(x, dim=1)
        return x.view(1, len(self.dlc_flags), -1), y


class ItchDetector(pl.LightningModule):
    def __init__(self, dataset_path, n, dlc_flags, num_train=15, num_val=10, val_dataset_path=None):
        """
        dataset_path : str - path to dataset pickle file
        n : int - frame radius (how many frames to look at to the left/right)
        dlc_flags | iterable - iterable with strings for each deeplabcut flag
        """
        super().__init__()

        self.dataset_path = dataset_path
        self.val_dataset_path = val_dataset_path
        self.auc = None
        self.n = n
        self.dlc_flags = dlc_flags
        self.num_train = num_train
        self.num_val = num_val
        self.thresholds = [i / 10.0 for i in range(11)]
        self.trainset = None
        self.valset = None

        # Define network architecture
        self.l1 = torch.nn.Conv2d(1, 1, kernel_size=(len(dlc_flags), (2 * n + 1)), padding=(0, n))

    def get_dataset(self):
        if self.val_dataset_path is None:
            self.main_dataset = ItchingDataset(self.dataset_path, self.n, self.dlc_flags)
            self.trainset, self.valset, _ = random_split(self.main_dataset, [self.num_train, self.num_val, len(
                self.main_dataset) - self.num_train - self.num_val])
        else:
            self.trainset = ItchingDataset(self.dataset_path, self.n, self.dlc_flags)

            self.valset = ItchingDataset(self.val_dataset_path, self.n, self.dlc_flags)
            # self.valset, _ = random_split(val_dataset, [self.num_val, len(val_dataset) - self.num_val])

    def forward(self, x):
        x = self.l1(x).mean(1).view(1, 1, -1)
        x = F.sigmoid(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'loss': F.binary_cross_entropy(y_hat, y)}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        pred_y = y_hat >= 0.5
        return {'loss': F.binary_cross_entropy(y_hat, y),
                'accuracy': torch.sum(y == pred_y).item() / y.numel()}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(1, 1, -1)
        y_hat = self(x)

        print(f'Number of Positives: {torch.sum(y == 1)}')
        print(f'Number of Network Positives: {torch.sum(y_hat > 0.5)}')

        positives = torch.sum(y == 1).item()
        negatives = np.prod(y.shape) - positives
        d = {'positives': positives, 'negatives': negatives}

        for threshold in self.thresholds:
            pred_y = y_hat >= threshold
            d[f'{threshold} true_positives'] = torch.sum((y == pred_y) * (y == 1)).item()
            d[f'{threshold} false_positives'] = torch.sum((y != pred_y) * (y == 0)).item()

        return d

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = np.mean([x['accuracy'] for x in outputs])
        tensorboard_logs = {'val_loss': avg_loss, 'accuracy': avg_acc}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def get_auc(self, true_positive_rates, false_positive_rates):
        area = 0
        for trial in range(1, len(self.thresholds)):
            average_trap_height = (true_positive_rates[trial] + true_positive_rates[trial - 1]) / 2
            trap_width = false_positive_rates[trial - 1] - false_positive_rates[trial]
            area += average_trap_height * trap_width
        return area

    def plot_roc(self, true_positive_rates, false_positive_rates):
        import matplotlib.pyplot as plt
        plt.plot(false_positive_rates, true_positive_rates)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('Receiver Operating Characteristic Curve')
        plt.savefig('roc.png')

    def test_epoch_end(self, outputs):
        true_positive_rates = []
        false_positive_rates = []

        for threshold in self.thresholds:
            true_positives = np.sum([output[f'{threshold} true_positives'] for output in outputs])
            total_positives = np.sum([output['positives'] for output in outputs])
            true_positive_rates.append(true_positives / total_positives)

            false_positives = np.sum([output[f'{threshold} false_positives'] for output in outputs])
            total_negatives = np.sum([output['negatives'] for output in outputs])
            false_positive_rates.append(false_positives / total_negatives)

        self.auc = self.get_auc(true_positive_rates, false_positive_rates)
        self.plot_roc(true_positive_rates, false_positive_rates)
        return {'log': {}}

    def train_dataloader(self):
        if self.trainset is None:
            self.get_dataset()
        return DataLoader(self.trainset, batch_size=1)

    def val_dataloader(self):
        if self.valset is None:
            self.get_dataset()
        return DataLoader(self.valset, batch_size=1)

    def test_dataloader(self):
        if self.valset is None:
            self.get_dataset()
        return DataLoader(self.valset, batch_size=1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


def plot_auc_trial(iv, dv, num_trials=2):
    import matplotlib.pyplot as plt
    plt.plot(iv, dv)
    plt.xlabel('Number of Training Videos')
    plt.ylabel('Area Under ROC Curve (AUC)')
    plt.title(f'Area Under ROC Curve (AUC) vs. Number of Training Videos')
    plt.savefig('auc_trial.png')
    plt.close()


def plot_model_weights(weights, dlc_flags):
    print(weights)
    import matplotlib.pyplot as plt

    for weight, dlc_flag in zip(weights, dlc_flags):
        plt.plot(list(range(len(weight))), weight, label=dlc_flag)
    plt.title(f'Weight Visualization')
    plt.xlabel('Frame #')
    plt.ylabel('Weight Magnitude')
    plt.legend()
    plt.savefig('abc.png')
    plt.close()


if __name__ == "__main__":
    from tensorify import *

    aucs = []
    for num_label_frames in [2500]:
        dlc_flags = ('leftpaw_prob', 'rightpaw_prob',)  # 'leftpaw_neck', 'rightpaw_neck')

        pickle.dump([create_batch('BW_MIT_200318_M6_R_3DeepCut_resnet50_mouse_behavior_idJan24shuffle1_200000.df.pkl',
                                  'BW_MIT_200318_R3_M6.human', frame_range=[0, num_label_frames],
                                  mult=30 / 29.981110061670094)],
                    open('human_label_dataset.pkl', 'wb'))
        pickle.dump([create_batch('BW_MIT_200318_M6_R_3DeepCut_resnet50_mouse_behavior_idJan24shuffle1_200000.df.pkl',
                                  'BW_MIT_200318_R3_M6.human', frame_range=[2500, 5000],
                                  mult=30 / 29.981110061670094)],
                    open('human_validation_dataset.pkl', 'wb'))

        model = ItchDetector('human_label_dataset.pkl', 5, dlc_flags,
                             num_train=1, val_dataset_path='human_validation_dataset.pkl')
        trainer = pl.Trainer(gpus=1, benchmark=True, max_epochs=3600)
        trainer.fit(model)
        # plot_model_weights(model.l1.weight.view(len(dlc_flags), -1).data.cpu().numpy(), dlc_flags)

        trainer.test(model)
        aucs.append((num_label_frames, model.auc))

    print(aucs)
# dlc_flags = ('leftpaw_prob', 'rightpaw_prob')

# num_samples_iter = list(frame_range(1, 3))
# num_trials = 2
#
# aucs = [] #processed_data.pkl

# plot_model_weights(model.l1.weight.view(len(dlc_flags), -1).data.cpu().numpy(), dlc_flags)
# num_train_aucs.append(model.auc)
# plot_auc_trial(num_samples_iter, aucs, num_trials=num_trials)
# print(aucs)
# print(num_samples_iter)
