import multiprocessing as mp
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from termcolor import colored

import mousenet as mn


def train_xgboost(X_train, y_train, X_val=None, y_val=None):
    data_dmatrix = xgb.DMatrix(data=X_train, label=y_train)
    clf = None
    if X_val is not None and y_val is not None:
        val_dmatrix = xgb.DMatrix(data=X_val, label=y_val)

        clf = xgb.train(dtrain=data_dmatrix,
                        params={'gpu_id': 1, 'tree_method': 'gpu_hist', 'objective': 'binary:logistic',
                                'nthread': mp.cpu_count(), },
                        num_boost_round=200, evals=[(val_dmatrix, 'val_set')], early_stopping_rounds=30,
                        feval=mn.prauc_feval, maximize=True)
    else:
        val_dmatrix = xgb.DMatrix(data=X_val, label=y_val)

        clf = xgb.train(dtrain=data_dmatrix,
                        params={'gpu_id': 1, 'tree_method': 'gpu_hist', 'objective': 'binary:logistic',
                                'nthread': mp.cpu_count(), }, num_boost_round=100)
    return clf


def generate_confusion_vector(clf, X_video_list, y_video_list, threshold=0.7):
    class_tensors = []
    y_preds = []
    for X, y in zip(X_video_list, y_video_list):
        y_pred_raw = clf.predict(xgb.DMatrix(data=X), ntree_limit=clf.best_ntree_limit)
        y_preds.append(y_pred_raw)
        y_pred_thresh = y_pred_raw > threshold
        class_tensor = mn.get_class_tensor(y_pred_thresh, y)
        class_tensors.append(class_tensor)
    return y_preds, class_tensors


def generate_drc(clf, video_map, cutoff=50000, pdf=None,
                 dlc_config_path='/home/pl/projects/pl/DLC/Retraining-BenR-2020-05-25/config.yaml',
                 video_folder='/home/pl/projects/pl/MWT/data/videos'):
    dlc = mn.DLCProject(config_path=dlc_config_path)
    videos = mn.folder_to_videos(video_folder, skip_words=('labeled', 'CFR'), labeled=True)
    dlc.infer_trajectories(videos)

    result = {}
    for video_num, video in enumerate(tqdm(videos)):

        video: mn.LabeledVideo = video
        try:
            treatment = video_map[video.get_name().split('.')[0]]
            if treatment == 'SKIP':
                continue

            y_pred = clf.predict(xgb.DMatrix(data=mn.video_to_dataset(video, window_size=100, df_map=mn.mouse_map)),
                                 ntree_limit=clf.best_ntree_limit)[:cutoff]
            y_pred_cumulative = np.cumsum(y_pred)

            if treatment not in result:
                result[treatment] = (1, y_pred_cumulative)
            else:
                prev_num, prev_mean = result[treatment]
                result[treatment] = prev_num + 1, ((prev_mean * prev_num) + y_pred_cumulative) / (prev_num + 1)
        except Exception as e:
            pass
    import matplotlib.pyplot as plt

    for treatment in result.keys():
        _, y_pred_cumulative = result[treatment]
        plt.plot(list(range(len(y_pred_cumulative))), y_pred_cumulative, label=treatment)

    plt.legend()
    if pdf is not None:
        pdf.savefig()
        plt.close()


# ben version
def evaluate_video(eval_video_path, clf, path_to_dlc_project, clf_force, dlc_force, cutoff=50000):
    import os

    dlc = mn.DLCProject(config_path=path_to_dlc_project)
    video_folder = '/home/pl/projects/pl/MWT/data/videos'
    videos = mn.folder_to_videos(video_folder, skip_words=('labeled', 'CFR'), labeled=True)
    dlc.infer_trajectories(videos)

    for video in videos:
        if os.path.samefile(video.path, eval_video_path):
            video: mn.LabeledVideo = video
            y_pred = clf.predict(xgb.DMatrix(data=mn.video_to_dataset(video, window_size=100, df_map=mn.mouse_map)),
                                 ntree_limit=clf.best_ntree_limit)
            events = y_pred
            break

    n_frames = events.size
    nom_fps = 30
    times = 1 / nom_fps * np.arange(0, n_frames)
    result = np.zeros((n_frames, 2))
    result[:, 0] = times
    result[:, 1] = events

    return result


def generate_prauc_vs_data(pdf=None,
                           dlc_config_path='/home/pl/projects/pl/DLC/Retraining-BenR-2020-05-25/config.yaml',
                           train_data_folder='/home/pl/projects/pl/MWT/data/behavioral_model_training',
                           json_label_file='/home/pl/projects/pl/MWT/human_label_json/2020-06-03_ben-synced.json'):
    dlc = mn.DLCProject(config_path=dlc_config_path)
    labeled_videos = mn.json_to_videos(train_data_folder, json_label_file)

    dlc.infer_trajectories(labeled_videos)

    X_train, X_val, y_train, y_val, X_video_list, y_video_list = mn.get_sklearn_dataset(labeled_videos,
                                                                                        window_size=100,
                                                                                        test_size=0.25,
                                                                                        df_map=mn.mouse_map)

    train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999]
    prauc = []
    for train_size in train_sizes:
        X_train_new, _, y_train_new, _ = train_test_split(X_train, y_train, train_size=train_size, shuffle=False)
        clf = train_xgboost(X_train_new, y_train_new, X_val, y_val)
        prauc.append(metrics.average_precision_score(y_val, clf.predict(xgb.DMatrix(data=X_val),
                                                                        ntree_limit=clf.best_ntree_limit)))

    plt.plot(train_sizes, prauc)
    plt.xlabel('Training Dataset Size')
    plt.ylabel('Average Precision/PRAUC')
    plt.title('Average Precision/PRAUC vs. Training Dataset Size')

    if pdf is not None:
        pdf.savefig()
        plt.close()


def generate_temporal_feature_ranking(clf, pdf=None):
    feat_imp = pd.Series(clf.get_fscore())

    def temporal_replace(str):
        return int('0' + ''.join([i for i in str if (i.isdigit())]))

    temporal_features = feat_imp.rename(axis='index', index=temporal_replace)
    resulttable_calc = temporal_features.groupby(temporal_features.index)
    group_result = resulttable_calc.agg(['mean']).sort_values(by='mean', ascending=False)

    plt.scatter(group_result.index, group_result)
    plt.ylabel('Feature Importance Score')
    plt.xlabel('Temporal Distance from Clasification Time (Frames)')
    plt.title('Feature Importance Score vs. Temporal Distance')
    if pdf is not None:
        pdf.savefig()
        plt.close()


def generate_feature_ranking(clf, pdf=None):
    feat_imp = pd.Series(clf.get_fscore())

    def feature_replace(str):
        return ''.join([i for i in str if not (i.isdigit() or i == '-')])

    features = feat_imp.rename(axis='index', index=feature_replace)
    resulttable_calc = features.groupby(features.index)
    group_result = resulttable_calc.agg(['mean']).sort_values(by='mean', ascending=False)
    group_result.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.xlabel('Feature Name')
    plt.tight_layout()
    if pdf is not None:
        pdf.savefig()
        plt.close()


class XGBoostClassifier:
    def __init__(self, path_to_clf_model):
        self.clf = pickle.load(open(path_to_clf_model, 'rb'))

    def infer_video(self, video, save_path, fps=30, force=False):
        save_path = save_path.format(VIDEO_ID=video.get_video_id())
        if force or not os.path.exists(save_path):
            y_pred = self.clf.predict(
                xgb.DMatrix(data=mn.video_to_dataset(video, window_size=100, df_map=mn.mouse_map)),
                ntree_limit=self.clf.best_ntree_limit)

            results = pd.DataFrame(data={'Event Confidence': y_pred}, index=np.arange(0, y_pred.size) / fps)
            results.to_pickle(save_path)

    def __call__(self, videos, save_path, fps=30, force=False):
        print(videos)
        for vid in tqdm(videos, desc='Running/Loading Inference'):
            self.infer_video(vid, save_path, fps, force)


def cluster_events(videos, save_path, thresh, eps=10, min_samples=2, force=False, ):
    for video in tqdm(videos, desc="Computing Clusters"):
        vid_save_path = save_path.format(VIDEO_ID=video.get_video_id())
        try:
            results: pd.DataFrame = pd.read_pickle(vid_save_path)
        except Exception as e:
            raise Exception(colored(f"Failed to load prediction csv @ {vid_save_path}. "
                                    "Did you run inference with the XGBoostClassifier?", 'red'))

        if force or 'Clustered Events' not in results.columns:
            all_points = np.array([range(len(results['Event Confidence']))]).swapaxes(0, 1)
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)

            results['Clustered Events'] = dbscan.fit_predict(all_points,
                                                             sample_weight=results['Event Confidence'] > thresh)
            results.to_pickle(vid_save_path)


def event_matching(human_label_file, machine_label_file, video_ids, save_path):
    stats = {
        "num_TP": 0,
        "num_FP": 0,
        "num_FN": 0,
    }

    # accounts for human delay and machine clustering only start point
    tolerance_sec = 10
    tolerance_min = tolerance_sec / 60

    for video_id in tqdm(video_ids, desc='Matching Events'):
        matching = dict()
        tp = []
        fn = []
        fp = []
        machine_matched = []
        human_matched = []

        # get results
        human_labelled_df = pd.read_pickle(human_label_file.format(VIDEO_ID=video_id.replace('CFR', '')))
        human_labelled = np.array([list(human_labelled_df.index.values), list(human_labelled_df['Event'])]).swapaxes(0,
                                                                                                                     1)
        machine_labelled_df = pd.read_pickle(machine_label_file.format(VIDEO_ID=video_id))
        machine_labelled_df = machine_labelled_df[machine_labelled_df["Clustered Events"] != -1]
        machine_labelled_df = machine_labelled_df[
            machine_labelled_df['Clustered Events'] != machine_labelled_df['Clustered Events'].shift(1)]
        machine_labelled = np.array(
            [list(machine_labelled_df.index.values), list(machine_labelled_df['Clustered Events'])]).swapaxes(0, 1)

        # convert
        machine_labelled = np.asarray(machine_labelled, dtype=np.float32)
        machine_labelled[:, 0] = machine_labelled[:, 0] / 60.0  # dim: [nevents x 2], units: [minutes x Z]

        # make graph
        dist = np.abs(human_labelled[:, 0][:, np.newaxis] - machine_labelled[:, 0])
        dist[dist > tolerance_min] = 1000
        human_event_assignments, machine_event_assignments = linear_sum_assignment(dist)

        for (human_event_assignment, machine_event_assignment) in zip(human_event_assignments,
                                                                      machine_event_assignments):
            if dist[human_event_assignment, machine_event_assignment] < tolerance_min:
                machine_matched.append(machine_labelled[machine_event_assignment, 1])
                human_matched.append(human_labelled[human_event_assignment, 1])
                tp.append(machine_labelled[machine_event_assignment, 0])

        for machine_time, machine_event in machine_labelled:
            if not machine_event in machine_matched:
                fp.append(machine_time)

        for human_time, human_event in human_labelled:
            if not human_event in human_matched:
                fn.append(human_time)

        stats["num_TP"] += len(tp)
        stats["num_FP"] += len(fp)
        stats["num_FN"] += len(fn)
        matching["TP"] = np.asarray(tp)
        matching["FP"] = np.asarray(fp)
        matching["FN"] = np.asarray(fn)
        stats[video_id] = matching

    stats["precision"] = stats["num_TP"] / (stats["num_TP"] + stats["num_FP"])
    stats["recall"] = stats["num_TP"] / (stats["num_TP"] + stats["num_FN"])
    print(f"Precision: {stats['precision']}")
    print(f"Recall: {stats['recall']}")
    import pickle
    pickle.dump(stats, open(save_path, 'wb'))
