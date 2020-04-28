import mousenet as mn
import torch
import logging
from pytorch_lightning import Trainer

logging.getLogger().setLevel(logging.DEBUG)  # Log all info

dlc = mn.DLCProject(config_path='/home/pl/sauhaarda/peptide_logic_refactored/dlc/'
                                'mouse_behavior_id-sauhaarda-2020-01-24/config.yaml')
labeled_videos = mn.json_to_videos('/home/pl/Data/mWT SR 017 (PL 100960 DRC IV)_renamed/', 'ben.json')

# # infer trajectories (if not already done) and assign trajectory files to video objects
dlc.infer_trajectories(labeled_videos)
#
#
# # Specify input to the network
# def df_map(df):
#     df['head', 'x'] = df['leftear']['x'] + df['rightear']['x']
#     df['head', 'y'] = df['leftear']['y'] + df['rightear']['y']
#     x = [mn.dist(df, 'neck', 'tail'), mn.dist(df, 'head', 'tail')]
#     return x
#
#
# dataset = mn.DLCDataset(labeled_videos, df_map)
#
# # Define Network Architecture
# logging.getLogger().setLevel(logging.ERROR)
#
#
# id = mn.ItchDetector(dataset, train_val_split=0.95)
# trainer = Trainer(gpus=1)
# trainer.fit(id)
