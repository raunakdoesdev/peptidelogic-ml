import glob
import json
import logging
import os
from tqdm import tqdm
import torch

from mousenet import util
import cv2
import pickle
import subprocess
import copy
import pandas as pd

class Video:
    def __init__(self, path):
        self.path = path
        self.num_frames = None

    def get_name(self):
        return os.path.basename(self.path)

    def get_num_frames(self):
        if self.num_frames is not None: return self.num_frames
        self.num_frames = int(cv2.VideoCapture(self.path).get(cv2.CAP_PROP_FRAME_COUNT))
        return self.num_frames - 1


class LabeledVideo(Video):
    def __init__(self, path):
        super().__init__(path)
        self.ground_truth = {}
        self.df_path = None
        self.labeled_video_path = None
        self.start = 0
        self.end = self.get_num_frames() - 1
        self.orig_start = None
        self.orig_end = None
        self.win_map = None
        self.cap = None
        self.df = None
        self.color_map = {}

    def grab_frame(self, frame_num):
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.path)

        if 'CFS' in self.path:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        else:
            self.get_windows_map()
            self.cap.set(cv2.CAP_PROP_POS_MSEC, self.win_map[frame_num])

        ret, frame = self.cap.read()
        return frame

    def grab_frame_with_bparts(self, frame_num, thresh=0.8):
        frame = self.grab_frame(frame_num)
        if self.df is None:
            self.df = pd.read_hdf(self.df_path)
            self.df = self.df[self.df.columns.get_level_values(0).unique()[0]]

        for key in self.df.columns.get_level_values(0):
            import random
            # self.color_map[key] = self.color_map.get(key, (random.randint(0, 255), random.randint(0, 255),
            #                                                random.randint(0, 255)))
            if self.df[key]['likelihood'][frame_num] > thresh:
                x, y = self.df[key]['x'][frame_num], self.df[key]['y'][frame_num]
                cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 0), 6)
        return frame

    def set_ground_truth(self, label_path, behavior):
        self.ground_truth[behavior] = label_path

    def set_df(self, df_path):
        self.df_path = df_path

    def get_windows_map(self, force=False):
        mapping_path = self.path.replace('mp4', 'winmap')
        if self.win_map is not None and not force:
            return
        elif os.path.exists(mapping_path) and not force:
            self.win_map = pickle.load(open(mapping_path, 'rb'))
        else:
            if os.name != 'nt':
                raise OSError('Need to create windows map on a windows system!')
            self.win_map = {}

            cap = cv2.VideoCapture(self.path)

            for read in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))),
                             desc=f'{self.path.split("/")[-1]} Read to Time Mapping'):
                _, image = cap.read()
                self.win_map[read] = cap.get(cv2.CAP_PROP_POS_MSEC)

            pickle.dump(self.win_map, open(mapping_path, 'wb'))

    def calculate_mappings(self, force=False):
        if not 'CFS' in self.path and not os.path.exists(self.path.replace(".mp4", "-CFS.mp4")) or True:
            return subprocess.Popen(
                f'ffmpeg -i "{self.path}" -r 30 -c:v libx264 -c:a copy -crf 0 "{self.path.replace(".mp4", "-CFS.mp4")}"',
                shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # mapping_path = self.path.replace('mp4', 'new_map')
        # if self.frame2read is not None and self.read2time is not None:
        #     return
        # elif os.path.exists(mapping_path) and not force:
        #     self.frame2read, self.read2time = pickle.load(open(mapping_path, 'rb'))
        # else:
        #     self.frame2read = {}
        #     self.read2time = {}
        #
        #     cap = cv2.VideoCapture(self.path)
        #
        #     for read in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))),
        #                      desc=f'{self.path.split("/")[-1]} Read to Time Mapping'):
        #         _, image = cap.read()
        #         self.read2time[read] = cap.get(cv2.CAP_PROP_POS_MSEC)
        #
        #     pickle.dump([self.frame2read, self.read2time], open(mapping_path, 'wb'))


def folder_to_videos(video_folder, skip_words=('labeled',), required_words=[], paths=False, labeled=False):
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

    for video_path in copy.deepcopy(video_paths):
        for skip_word in skip_words:
            if skip_word in video_path:
                video_paths.remove(video_path)
                break

    for video_path in copy.deepcopy(video_paths):
        for required_word in required_words:
            if required_word not in video_path:
                video_paths.remove(video_path)
                break
    if labeled:
        return video_paths if paths else [LabeledVideo(video_path) for video_path in video_paths]
    return video_paths if paths else [Video(video_path) for video_path in video_paths]


def json_to_videos(video_folder, json_path, mult=1):
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
            frames = list(range(int(labels['Label Start']), int(labels['Label End'])))
            ground_truth = torch.FloatTensor([util.in_range(labels['event_ranges'], frame) for frame in frames])

            ground_truth_path = video.path.split('.mp4')[0] + f'{behavior}.lbl'
            torch.save(ground_truth, ground_truth_path)
            video.set_ground_truth(ground_truth_path, behavior)
            video.start, video.end = round(labels['Label Start']), round(labels['Label End'])

    return video_list
