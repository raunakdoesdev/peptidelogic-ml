import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.absolute().parent.absolute()))

import mousenet as mn
dlc = mn.DLCProject(config_path='/home/pl/Retraining-BenR-2020-05-25/config.yaml')
videos = mn.folder_to_videos('/home/pl/Data/mWT SR 017 (PL 100960 DRC IV)_renamed', labeled=True, required_words=('CFS',))
for video in videos:
    print(video.path)
dlc.infer_trajectories(videos)
