import numpy as np
from sklearn.model_selection import train_test_split

X, y = np.arange(10).reshape((5, 2)), range(5)

# import logging
# import mousenet as mn
# from tqdm import tqdm
#
# logging.getLogger().setLevel(logging.DEBUG)  # Log all info
# videos = mn.folder_to_videos(r'E:\Peptide Logic\Writhing', labeled=True)
#
# processes = []
# for labeled_video in videos:
#     labeled_video.get_windows_map()
