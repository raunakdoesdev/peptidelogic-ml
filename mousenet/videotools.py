import glob
import json
import logging
import os
from tqdm import tqdm
import torch

from mousenet import util
import cv2
import pickle


class Video:
    def __init__(self, path):
        self.path = path

    def get_name(self):
        return os.path.basename(self.path)

    def get_num_frames(self):
        return cv2.VideoCapture(self.path).get(cv2.CAP_PROP_FRAME_COUNT)


class LabeledVideo(Video):
    def __init__(self, path):
        super().__init__(path)
        self.ground_truth = {}
        self.df_path = None
        self.labeled_video_path = None
        self.start = None
        self.end = None
        self.orig_start = None
        self.orig_end = None
        self.frame2read = None
        self.read2time = None

    def set_ground_truth(self, label_path, behavior):
        self.ground_truth[behavior] = label_path

    def set_df(self, df_path):
        self.df_path = df_path

    def calculate_mappings(self, force=False):
        pass
        mapping_path = self.path.replace('mp4', 'map')
        if self.frame2read is not None and self.read2time is not None:
            return
        elif os.path.exists(mapping_path) and not force:
            self.frame2read, self.read2time = pickle.load(open(mapping_path, 'rb'))
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
        #     for read in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))),
        #                      desc=f'{self.path.split("/")[-1]} Frame to Read Mapping'):
        #         _, image = cap.read()
        #         cap.set(cv2.CAP_PROP_POS_MSEC, self.read2time[read])
        #         self.frame2read[int(cap.get(cv2.CAP_PROP_POS_FRAMES))] = read
        #
        #         if read == 1000:
        #             print(self.frame2read)
        #
        #     pickle.dump([self.frame2read, self.read2time], open(mapping_path, 'wb'))
        #
        # for i in range(list(self.frame2read.keys())[-1]):
        #     x = self.frame2read.get(i)
        #     if not x:
        #         self.frame2read[i] = self.frame2read[i - 1]

    def frame_to_read(self, frame):
        # if abs(self.frame2read[int(frame)] - int(frame)) >= 2:
        #     print(f"THIS IS WORKING! {frame} -> {self.frame2read[int(frame)]}")
        return int(frame)


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
        video.calculate_mappings()  # get mappings for each video
        for behavior in human_label[video.get_name()].keys():
            labels = human_label[video.get_name()][behavior]
            frames = list(range(video.frame_to_read(labels['Label Start']), video.frame_to_read(labels['Label End'])))
            ground_truth = torch.FloatTensor([util.in_range(labels['event_ranges'],
                                                            frame, map=video.frame_to_read) for frame in frames])

            ground_truth_path = video.path.split('.mp4')[0] + f'{behavior}.lbl'
            torch.save(ground_truth, ground_truth_path)
            video.set_ground_truth(ground_truth_path, behavior)
            video.start, video.end = round(video.frame_to_read(labels['Label Start'])), \
                                     round(video.frame_to_read(labels['Label End']))

            video.orig_start, video.orig_end = round(labels['Label Start']), round(labels['Label End'])
    return video_list
