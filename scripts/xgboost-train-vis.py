import os
import pickle

import sys 
sys.path.insert(0, "../")

import mousenet as mn

dlc = mn.DLCProject(config_path='/home/pl/projects/pl/DLC/Retraining-BenR-2020-05-25/config.yaml')
labeled_videos = mn.json_to_videos('/home/pl/projects/pl/MWT/data/behavioral_model_training',
                                   '/home/pl/projects/pl/MWT/human_label_json/2020-06-03_ben-synced.json')

dlc.infer_trajectories(labeled_videos)

X_train, X_test, y_train, y_test, X_video_list, y_video_list = mn.get_sklearn_dataset(labeled_videos, window_size=100,
                                                                                      test_size=0.25,
                                                                                      df_map=mn.mouse_map)

model_path = '/home/pl/projects/pl/MWT/models/xgb.save'
force = True
if force or not os.path.exists(model_path):
    clf = mn.train_xgboost(X_train, y_train, X_test, y_test)
    pickle.dump(clf, open(model_path, 'wb'))
else:
    clf = pickle.load(open(model_path, 'rb'))

y_preds, class_tensors = mn.generate_confusion_vector(clf, X_video_list, y_video_list)
mn.visual_debugger(labeled_videos, y_preds, y_video_list, class_tensors, val_percent=0.25)
