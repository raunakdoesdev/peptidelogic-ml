import logging

import mousenet as mn

logging.getLogger().setLevel(logging.DEBUG)  # Log all info

# Setup DLC Project
dlc = mn.DLCProject(config_path='/home/pl/sauhaarda/peptide_logic_refactored/dlc/'
                                'mouse_behavior_id-sauhaarda-2020-01-24/config.yaml', pcutoff=0.25)

videos = mn.folder_to_videos('/home/pl/Data/mWT SR 017 (PL 100960 DRC IV)', labeled=True, required_words=['199107'])
dlc.infer_trajectories(videos)
for video in videos:
    video.calculate_mappings()