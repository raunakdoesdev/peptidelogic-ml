import logging
import mousenet as mn
from tqdm import tqdm

logging.getLogger().setLevel(logging.DEBUG)  # Log all info
videos = mn.folder_to_videos('/home/pl/Data/mWT SR 017 (PL 100960 DRC IV)_renamed', labeled=True)

processes = []
for labeled_video in videos:
    processes.append(labeled_video.calculate_mappings())

for p in processes: p.wait()
