import pickle
from collections import defaultdict

import toml
import json

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
    if cfg['training']['dlc']:
        dlc = mn.DLCProject(config_path=cfg['inference']['classifier']['models']['dlc_config_path'])
        dlc.create_training_set()
        dlc.train_network()

    if cfg['training']['xgb']['refresh']:
        dlc = mn.DLCProject(config_path=cfg['inference']['classifier']['models']['dlc_config_path'])
        labeled_videos = mn.json_to_videos(cfg['videos']['path_to_videos'],
                                           cfg['training']['xgb']['label_json'])
        dlc.infer_trajectories(labeled_videos, force=cfg['inference']['classifier']['force_dlc'])

        print('Generating Dataset. This may take a while...')
        X_train, X_test, y_train, y_test, X_video_list, y_video_list = mn.get_sklearn_dataset(labeled_videos,
                                                                                              window_size=100,
                                                                                              test_size=cfg['training']['xgb']['test_size'],
                                                                                              df_map=mn.mouse_map)

        clf = mn.train_xgboost(X_train, y_train, X_test, y_test)
        pickle.dump(clf, open(cfg['training']['xgb']['save_path'], 'wb'))
        cfg['inference']['classifier']['models']['xgb_model_path'] = cfg['training']['xgb']['save_path']  # if training, then use

    if cfg['inference']['classifier']['refresh']:
        dlc = mn.DLCProject(config_path=cfg['inference']['classifier']['models']['dlc_config_path'])
        dlc.infer_trajectories(videos, force=cfg['inference']['classifier']['force_dlc'])
        xgb = mn.XGBoostClassifier(cfg['inference']['classifier']['models']['xgb_model_path'])
        xgb(videos, cfg['inference']['save_path'], force=cfg['inference']['classifier']['force_xgb'])

    if cfg['inference']['cluster']['refresh']:
        mn.cluster_events(videos, cfg['inference']['save_path'], eps=cfg['inference']['cluster']['eps'],
                          min_samples=cfg['inference']['cluster']['min_samples'],
                          force=cfg['inference']['cluster']['force'], thresh=cfg['inference']['cluster']['thresh'])

    if cfg['human_labels']['refresh']:
        mn.extract_human_labels(load(cfg['human_labels']['blind_key_to_video_id']),
                                load(cfg['human_labels']['individual_files']),
                                cfg['human_labels']['save_path'])

    if cfg['event_matching']['refresh']:
        mn.event_matching(cfg['human_labels']['save_path'], cfg['inference']['save_path'], cfg['videos']['video_ids'],
                          cfg['event_matching']['save_path'])

    if cfg['visual_debugger']['run']:
        cfr_videos = [video for video in videos if 'CFR' in video.get_video_id()]
        y_pred, y_clustered = mn.get_ypreds(cfr_videos, cfg['inference']['save_path'])
        human_pred = mn.get_human_pred(cfr_videos, cfg['human_labels']['save_path'])
        class_vectors = mn.generate_classification_vector(cfr_videos, cfg['event_matching']['save_path'])
        mn.visual_debugger(cfr_videos, (y_clustered, y_pred, human_pred), class_vectors, val_percent=0)

    if cfg['visualization']['refresh']:
        if cfg['visualization']['video_instances']:
            if cfg['visualization']['plot_matching']:
                matching_stats = pickle.load(open(cfg['event_matching']['save_path'], 'rb'))
            else:
                matching_stats = defaultdict(lambda: None)

            human_labels = mn.extract_video_human_labels(cfg['videos']['video_ids'],
                                                         cfg['human_labels']['summary_file'],
                                                         cfg['human_labels']['blind_key_to_video_id'],
                                                         cfg['videos']['dosage'])

            print(human_labels.keys())
            for video in tqdm(videos, desc='Plotting Single Videos'):
                video_id = video.get_video_id()
                mn.vis.plot_single_video_instance(human_labels[video_id.replace('CFR', '')],
                                                  cfg['inference']['save_path'],
                                                  video_id, matching_stats[video_id])

        if cfg['visualization']['drc']:
            dosages = mn.get_dose_to_video_ids(cfg['videos']['dosage'])
            human_labels = mn.extract_drc_human_labels(cfg['videos']['video_ids'], cfg['human_labels']['summary_file'],
                                                       cfg['human_labels']['blind_key_to_video_id'],
                                                       cfg['videos']['dosage'])
            mn.vis.plot_drc(human_labels, cfg['inference']['save_path'], dosages)

        mn.vis.save_figs(cfg['visualization']['save_path'])

        if cfg['visualization']['auto_open']:
            mn.vis.open_figs(cfg['visualization']['save_path'])


if __name__ == '__main__':
    config = toml.load('config.toml')
    main(config)
