import logging
import mousenet as mn
import torch

logging.getLogger().setLevel(logging.DEBUG)  # Log all info

dlc = mn.DLCProject(config_path='/home/pl/sauhaarda/peptide_logic_refactored/dlc/'
                                'mouse_behavior_id-sauhaarda-2020-01-24/config.yaml')
labeled_videos = mn.json_to_videos('/home/pl/Data/mWT SR 017 (PL 100960 DRC IV)_renamed/', 'benv2.json',
                                   mult=30 / 29.981110061670094)

# Infer trajectories
dlc.infer_trajectories(labeled_videos)

# Define hyperparameters
hparams = {'num_filters': (7, (1, 20)),
           'num_filters2': (7, (1, 20)),
           'filter_width': (851, (601, 1001, 50)),  # must be an odd number
           'filter_width2': (801, (201, 1001, 50)),  # must be an odd number
           'in_channels': 6,  # number of network inputs
           'weight': 7,  # how much "emphasis" to give to positive labels
           'loss': torch.nn.functional.binary_cross_entropy,
           'train_val_split': 0.75}


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
dataset = mn.DLCDataset(labeled_videos, df_map)
runner = mn.Runner(MouseModel, hparams, dataset)
model, auc = runner.train_model()

# runner.hyperparemeter_optimization(timeout=600, n_trials=None)
model.eval()
model.cpu()
print(model(dataset[0][0]).detach())

dlc.create_labeled_videos(labeled_videos)
mn.VisualDebugger(div=30 / 29.981110061670094)
