import glob
import json
import logging
import os

import torch

from mousenet import util


class Video:
    def __init__(self, path):
        self.path = path

    def get_name(self):
        return os.path.basename(self.path)


class LabeledVideo(Video):
    def __init__(self, path):
        super().__init__(path)
        self.ground_truth = {}
        self.df_path = None
        self.labeled_video_path = None
        self.start = None
        self.end = None

    def set_ground_truth(self, label_path, behavior):
        self.ground_truth[behavior] = label_path

    def set_df(self, df_path):
        self.df_path = df_path


def folder_to_videos(video_folder, skip_words=('filtered_labeled',), paths=False):
    """
    Returns list of video objects from given folder.
    :param video_folder:
    :param skip_words:
    :param paths: optional, return paths instead of video object
    :return:
    """
    if video_folder is None or not os.path.exists(video_folder):
        raise IOError('Invalid video folder input!')

    video_paths = glob.glob(os.path.join(video_folder, "ASLKDFJSLD.mp4").replace('ASLKDFJSLD', '/**/*'))
    video_paths += glob.glob(os.path.join(video_folder, "ASLKDFJSLD.mp4").replace('ASLKDFJSLD', '/*'))

    for video_path in video_paths:
        for skip_word in skip_words:
            if skip_word in video_path:
                video_paths.remove(video_path)
                break

    return video_paths if paths else [Video(video_path) for video_path in video_paths]


def json_to_videos(video_folder, json_path, mult):
    """
    Creates video object with ground_truth for all annotated videos in the provided JSON file.
    :param video_folder:
    :param json_path:
    :return: list of video objects with ground_truth
    """
    human_label = json.load(open(json_path, 'r'))
    videos = folder_to_videos(video_folder, paths=True)
    video_list = []
    for key in human_label.keys():
        if not key.endswith('.mp4'): continue
        no_match = True
        for video in videos:
            if key in video:
                video_list.append(LabeledVideo(video))
                logging.info(f'{key} found at {video}')
                no_match = False
                break
        if no_match:
            logging.warning(f'{key} not found in {video_folder} Skipping this file.')

    for video in video_list:
        for behavior in human_label[video.get_name()].keys():
            labels = human_label[video.get_name()][behavior]
            frames = list(range(round(labels['Label Start'] * mult), round(labels['Label End'] * mult)))
            ground_truth = torch.FloatTensor([util.in_range(labels['event_ranges'], frame, mult) for frame in frames])
            ground_truth_path = video.path.split('.mp4')[0] + f'{behavior}.lbl'
            torch.save(ground_truth, ground_truth_path)
            video.set_ground_truth(ground_truth_path, behavior)
            video.start, video.end = round(labels['Label Start'] * mult), round(labels['Label End'] * mult)

    return video_list
