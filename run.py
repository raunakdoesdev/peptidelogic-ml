import pickle
from collections import defaultdict

import toml
import json

from sklearn import metrics
from tqdm import tqdm

import mousenet as mn


def load(x):
    """
    Load json if path to json. Otherwise, returns input.
    """
    if type(x) == str and '.json' in x:
        return json.load(open(x, 'r'))
    return x


def main(cfg):
    videos = mn.ids_to_videos(cfg['videos']['path_to_videos'], load(cfg['videos']['video_ids']))

    if cfg['dlc']['retrain']:
        dlc = mn.DLCProject(config_path=cfg['dlc']['config_path'])
        dlc.create_training_set()
        dlc.train_network()

    if cfg['xgb']['retrain']:
        dlc = mn.DLCProject(config_path=cfg['dlc']['config_path'])
        labeled_videos = mn.json_to_videos(cfg['videos']['path_to_videos'],
                                           cfg['xgb']['labels'])

        dlc.infer_trajectories(labeled_videos, force=cfg['dlc']['reinfer'])
        X_train, X_test, y_train, y_test, X_video_list, y_video_list = \
            mn.get_sklearn_dataset(labeled_videos, window_size=cfg['xgb']['window_size'],
                                   test_size=cfg['xgb']['test_size'], df_map=mn.mouse_map)

        clf = mn.train_xgboost(X_train, y_train, X_test, y_test)
        pickle.dump(clf, open(cfg['xgb']['model_path'], 'wb'))

    if cfg['xgb']['reinfer'] or cfg['xgb']['retrain']:
        dlc = mn.DLCProject(config_path=cfg['dlc']['config_path'])
        dlc.infer_trajectories(videos, force=cfg['dlc']['reinfer'])
        xgb = mn.XGBoostClassifier(cfg['xgb']['model_path'])
        xgb(videos, cfg['cache_paths']['machine_label'], force=True)

    if cfg['clustering']['redo'] or cfg['xgb']['reinfer'] or cfg['xgb']['retrain']:
        mn.cluster_events(videos, cfg['cache_paths']['machine_label'], eps=cfg['clustering']['eps'],
                          min_samples=cfg['clustering']['min_samples'],
                          force=True, thresh=cfg['clustering']['thresh'])

    if cfg['human_labels']['refresh']:
        mn.extract_human_labels(load(cfg['human_labels']['blind_key_to_video_id']),
                                load(cfg['human_labels']['individual_files']),
                                cfg['cache_paths']['human_label'])

    if cfg['event_matching']['run']:
        mn.event_matching(cfg['cache_paths']['human_label'], cfg['cache_paths']['machine_label'],
                          cfg['videos']['video_ids'], cfg['cache_paths']['event_matching'])


    if cfg['visual_debugger']['train']:
        dlc = mn.DLCProject(config_path=cfg['dlc']['config_path'])
        labeled_videos = mn.json_to_videos(cfg['videos']['path_to_videos'], cfg['xgb']['labels'])
        dlc.infer_trajectories(labeled_videos, force=cfg['dlc']['reinfer'])
        clf = mn.XGBoostClassifier(cfg['xgb']['model_path']).clf
        X_train, X_test, y_train, y_test, X_video_list, y_video_list = \
            mn.get_sklearn_dataset(labeled_videos, window_size=cfg['xgb']['window_size'],
                                   test_size=cfg['xgb']['test_size'], df_map=mn.mouse_map)
        y_preds, class_tensors = mn.generate_confusion_vector(clf, X_video_list, y_video_list)
        mn.visual_debugger(labeled_videos, [y_preds, y_video_list], class_tensors, val_percent=0.25)

    if cfg['visual_debugger']['infer']:
        dlc = mn.DLCProject(config_path=cfg['dlc']['config_path'])
        cfr_videos = [video for video in videos if 'CFR' in video.get_video_id()]
        dlc.infer_trajectories(cfr_videos, force=cfg['dlc']['reinfer'])
        y_pred, y_clustered = mn.get_ypreds(cfr_videos, cfg['cache_paths']['machine_label'])
        human_pred = mn.get_human_pred(cfr_videos, cfg['cache_paths']['human_label'])
        class_vectors = mn.generate_classification_vector(cfr_videos, cfg['cache_paths']['event_matching'])
        mn.visual_debugger(cfr_videos, (y_clustered, y_pred, human_pred), class_vectors, val_percent=0)


    if cfg['visualization']['refresh']:
        if cfg['visualization']['pr_auc_vs_data']:
            import xgboost
            dlc = mn.DLCProject(config_path=cfg['dlc']['config_path'])
            labeled_videos = mn.json_to_videos(cfg['videos']['path_to_videos'],
                                               cfg['xgb']['labels'])
            dlc.infer_trajectories(labeled_videos, force=cfg['dlc']['reinfer'])
            X_train, X_test, y_train, y_test, X_video_list, y_video_list = \
                mn.get_sklearn_dataset(labeled_videos, window_size=cfg['xgb']['window_size'],
                                       test_size=cfg['xgb']['test_size'], df_map=mn.mouse_map)
            mn.prauc_vs_data(X_train, X_test, y_train, y_test)

        if cfg['visualization']['video_instances']:
            if cfg['visualization']['plot_matching']:
                matching_stats = pickle.load(open(cfg['cache_paths']['event_matching'], 'rb'))
            else:
                matching_stats = defaultdict(lambda: None)

            human_labels = mn.extract_video_human_labels(cfg['videos']['video_ids'],
                                                         cfg['human_labels']['summary_file'],
                                                         cfg['human_labels']['blind_key_to_video_id'],
                                                         cfg['videos']['dosage'])
            max_events = 0
            import pandas as pd
            import numpy as np
            for video in videos:
                video_id = video.get_video_id()
                df_path = cfg['cache_paths']['machine_label'].format(VIDEO_ID=video_id)
                max_events = max(max_events, pd.read_pickle(df_path)['Clustered Events'].max())
                max_events = max(max_events, np.mean([sum(h[:, 1]) for h in human_labels[video_id.replace('CFR', '')]]))
            for video in tqdm(videos, desc='Plotting Single Videos'):
                video_id = video.get_video_id()
                mn.vis.plot_single_video_instance(human_labels[video_id.replace('CFR', '')],
                                                  cfg['cache_paths']['machine_label'],
                                                  video_id, matching_stats[video_id], y_top=max_events+10)

        if cfg['visualization']['drc']:
            dosages = mn.get_dose_to_video_ids(cfg['videos']['dosage'])
            human_labels = mn.extract_drc_human_labels(cfg['videos']['video_ids'], cfg['human_labels']['summary_file'],
                                                       cfg['human_labels']['blind_key_to_video_id'],
                                                       cfg['videos']['dosage'])
            mn.vis.plot_drc(human_labels, cfg['cache_paths']['machine_label'], dosages)

        mn.vis.save_figs(cfg['visualization']['save_path'])

        if cfg['visualization']['auto_open']:
            mn.vis.open_figs(cfg['visualization']['save_path'])


if __name__ == '__main__':
    config = toml.load('config.toml')
    main(config)
