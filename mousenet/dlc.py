import contextlib
import multiprocessing as mp
import os
import warnings
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import logging
from mousenet import util

warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ["DLClight"] = "True"
with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
    import deeplabcut


class DLCProject:
    def __init__(self, config_path=None, proj_name=None, labeler_name=None, videos=None, bodyparts=None, proj_dir='.',
                 numframespervid=20, pcutoff=0.9, cfg_update={}):
        if config_path is not None:
            self.config_path = os.path.abspath(config_path)
        elif proj_name is not None and labeler_name is not None and videos is not None and bodyparts is not None:
            self.create_project(proj_name=proj_name, labeler_name=labeler_name, videos=videos, bodyparts=bodyparts,
                                proj_dir=proj_dir, numframespervid=numframespervid, pcutoff=pcutoff,
                                cfg_update=cfg_update)
        else:
            raise Exception("Invalid parameters provided for initialization."
                            "Either provide config path or all other params.")

    def create_project(self, proj_name, labeler_name, videos, bodyparts, proj_dir, numframespervid, pcutoff,
                       cfg_update):
        print(os.path.abspath(proj_dir))

        try:
            self.config_path = os.path.abspath(deeplabcut.create_new_project(
                proj_name,
                labeler_name,
                [video.path for video in videos],
                working_directory=os.path.abspath(proj_dir)
            ))
        except Exception as e:
            raise Exception(f"{e}\nError constructing project. Please try again.")

        if self.config_path is None:
            raise NameError("Project already exists. Please try again.")

        cfg_update.update({'bodyparts': bodyparts, 'numframes2pick': numframespervid, 'pcutoff': pcutoff})
        self.config_update(cfg_update)

    def config_update(self, update):
        cfg = deeplabcut.auxiliaryfunctions.read_config(self.config_path)

        for k, v in update.items():
            cfg[k] = v

        deeplabcut.auxiliaryfunctions.write_config(self.config_path, cfg)

    def extract_frames(self, config_path, manual=False):
        if manual:
            deeplabcut.extract_frames(self.config_path, 'manual', userfeedback=False,
                                      slider_width=60)
        else:
            deeplabcut.extract_frames(self.config_path)

    def create_training_set(self, num_shuffles=1):
        deeplabcut.create_training_dataset(self.config_path, num_shuffles=num_shuffles)

    def train_network(self, shuffle=1, gpu=0):
        deeplabcut.train_network(self.config_path, shuffle=shuffle, gputouse=gpu)

    def get_scorer_name(self, shuffle=1, trainingsetindex=0):
        """
        Stupid helper function that should be implemented in DLC but it's not!!!
        :param shuffle:
        :param trainingsetindex:
        :return:
        """
        cfg = deeplabcut.auxiliaryfunctions.read_config(self.config_path)
        train_fraction = cfg['TrainingFraction'][trainingsetindex]
        model_folder = os.path.join(cfg["project_path"],
                                    str(deeplabcut.auxiliaryfunctions.GetModelFolder(train_fraction, shuffle, cfg)))
        path_test_config = Path(model_folder) / 'test' / 'pose_cfg.yaml'
        try:
            dlc_cfg = deeplabcut.pose_estimation_tensorflow.load_config(str(path_test_config))
        except FileNotFoundError:
            raise FileNotFoundError(
                "It seems the model for shuffle %s and trainFraction %s does not exist." % (shuffle, train_fraction))

        modelfolder = os.path.join(cfg["project_path"],
                                   str(deeplabcut.auxiliaryfunctions.GetModelFolder(train_fraction, shuffle, cfg)))

        # Check which snapshots are available and sort them by # iterations
        try:
            Snapshots = np.array(
                [fn.split('.')[0] for fn in os.listdir(os.path.join(modelfolder, 'train')) if "index" in fn])
        except FileNotFoundError:
            raise FileNotFoundError(
                "Snapshots not found! It seems the dataset for shuffle %s has not been trained/does not exist.\n Please train it before using it to analyze videos.\n Use the function 'train_network' to train the network for shuffle %s." % (
                    shuffle, shuffle))

        if cfg['snapshotindex'] == 'all':
            print(
                "Snapshotindex is set to 'all' in the config.yaml file. Running video analysis with all snapshots is very costly! Use the function 'evaluate_network' to choose the best the snapshot. For now, changing snapshot index to -1!")
            snapshotindex = -1
        else:
            snapshotindex = cfg['snapshotindex']

        increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
        Snapshots = Snapshots[increasing_indices]

        ##################################################
        # Load and setup CNN part detector
        ##################################################

        # Check if data already was generated:
        dlc_cfg['init_weights'] = os.path.join(modelfolder, 'train', Snapshots[snapshotindex])
        trainingsiterations = (dlc_cfg['init_weights'].split(os.sep)[-1]).split('-')[-1]
        return deeplabcut.auxiliaryfunctions.GetScorerName(cfg, shuffle, train_fraction,
                                                           trainingsiterations=trainingsiterations)

    def infer_trajectories(self, videos, num_procs=None, infer=True):
        uninferred_video_paths = []

        with util.DisableLogger():
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                scorer = self.get_scorer_name()

        for video in videos:
            label_path = f"{video.path.split('.mp4')[0]}{scorer}.h5"
            video.set_df(label_path)
            if not os.path.exists(label_path):
                uninferred_video_paths.append(video.path)
            else:
                logging.warning(f'Label file exists @ {label_path}. Skipping inference on {video.path}')

        if len(uninferred_video_paths) > 0:
            with util.DisableLogger():
                with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                    num_gpus = 4
                    num_procs = num_gpus if num_procs is None else num_procs
                    pool = mp.Pool(num_procs)
                    chunked_video_paths = np.array_split(uninferred_video_paths, num_procs)

                    for proc in range(num_procs):
                        pool.apply_async(deeplabcut.analyze_videos, (self.config_path, chunked_video_paths[proc],),
                                         {'save_as_csv': True, 'gputouse': proc % num_gpus})

                    pool.close()
                    pool.join()

    def create_labeled_videos(self, videos, num_procs=None, infer=True):
        todo_video_paths = []

        with util.DisableLogger():
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                scorer = self.get_scorer_name()

        for video in videos:
            labeled_video_path = f"{video.path.split('.mp4')[0]}{scorer}filtered_labeled.mp4"
            video.labeled_video_path = labeled_video_path
            if not os.path.exists(labeled_video_path):
                todo_video_paths.append(video.path)
            else:
                logging.warning(
                    f'Video file exists @ {labeled_video_path}. Skipping labeled video creation on {video.path}')

        if len(todo_video_paths) > 0:
            with util.DisableLogger():
                with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                    num_procs = mp.cpu_count()
                    pool = mp.Pool(num_procs)
                    chunked_video_paths = np.array_split(todo_video_paths, num_procs)

                    for proc in range(num_procs):
                        pool.apply_async(deeplabcut.create_labeled_video, (self.config_path, chunked_video_paths[proc]))

                    pool.close()
                    pool.join()
