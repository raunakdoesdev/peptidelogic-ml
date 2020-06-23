import mousenet as mn
import torch
import os


# Define Network Input
def df_map(df):
    # x = [mn.dist(df, 'leftpaw', 'tail'), mn.dist(df, 'rightpaw', 'tail'), mn.dist(df, 'neck', 'tail'), body_length,
    #      df['leftpaw']['likelihood'], df['rightpaw']['likelihood']]
    x = []
    for col in df.columns:
        x.append(df[col[0]][col[1]])
    # x = [df['hindpaw_right']['likelihood'], df['hindpaw_left']['likelihood'], df['hindheel_right']['likelihood'],
    #      df['hindheel_left']['likelihood'], df['frontpaw_left']['likelihood'], df['frontpaw_right']['likelihood'],
    #      mn.dist(df, 'tail', 'hindpaw_right'), mn.dist(df, 'tail', 'hindpaw_left')]
    return x


# Define hyperparameters
writhing_hparams = {'num_filters': (16, (1, 20)),
                    'num_filters2': (16, (1, 20)),
                    'filter_width': (21, (11, 101, 10)),  # must be an odd number
                    'filter_width2': (31, (11, 101, 10)),  # must be an odd number
                    'in_channels': 30,  # number of network inputs
                    'weight': (9, (1, 20)),  # how much "emphasis" to give to positive labels
                    'loss': torch.nn.functional.binary_cross_entropy,
                    'percent_data': None}


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


def train_writhing_network(force=False):
    torch.manual_seed(0)  # consistent behavior w/ random seed
    model_path = 'writhing.model'

    dlc = mn.DLCProject(config_path='/home/pl/Retraining-BenR-2020-05-25/config.yaml')
    hparams = writhing_hparams
    labeled_videos = mn.json_to_videos('/home/pl/Data', '../2020-06-03_ben-synced.json', mult=1)
    dlc.infer_trajectories(labeled_videos)
    dataset = mn.DLCDataset(labeled_videos, df_map, behavior='Writhe')
    runner = mn.Runner(MouseModel, hparams, dataset)
    if os.path.exists(model_path) and not force:
        print("HELLO")
        return runner.get_model(torch.load(model_path)).cuda()
    else:
        import numpy as np
        results = []
        trials = np.array(list(range(1, 10))) / 10.0
        print(trials)
        for percent_data in [None]:
            hparams['percent_data'] = percent_data
            runner = mn.Runner(MouseModel, hparams, dataset)
            runner.train_model(max_epochs=500)
            runner.trainer.test()
            results.append(runner.lightning_module.test_log['average_precision'])
        print(results)


if __name__ == '__main__':
    train_writhing_network(force=True)
