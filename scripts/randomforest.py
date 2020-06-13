import os
import pickle
from sklearn import metrics
import multiprocessing as mp
import mousenet as mn
import xgboost as xgb

model_path = 'XGB.save'
force = False
dlc = mn.DLCProject(config_path='/home/pl/Retraining-BenR-2020-05-25/config.yaml')
labeled_videos = mn.json_to_videos('/home/pl/Data', '../2020-06-03_ben-synced.json', mult=1)
dlc.infer_trajectories(labeled_videos)
X_train, X_test, y_train, y_test, X_video_list, y_video_list = mn.get_sklearn_dataset(labeled_videos, window_size=100,
                                                                                      test_size=0.25)

if force or not os.path.exists(model_path):
    data_dmatrix = xgb.DMatrix(data=X_train, label=y_train)
    val_dmatrix = xgb.DMatrix(data=X_test, label=y_test)

    clf = xgb.train(dtrain=data_dmatrix, params={'objective': 'binary:logistic', 'nthread': mp.cpu_count(), },
                    num_boost_round=200, evals=[(val_dmatrix, 'val_set')], early_stopping_rounds=30,
                    feval=mn.prauc_feval, maximize=True)
    pickle.dump(clf, open(model_path, 'wb'))
else:
    clf = pickle.load(open(model_path, 'rb'))

y_pred_raw = clf.predict(xgb.DMatrix(data=X_test))
print('Average Precision', metrics.average_precision_score(y_test, y_pred_raw))
print("AUC", metrics.roc_auc_score(y_test, y_pred_raw))

threshold = 0.7

class_tensors = []
y_preds = []
for X, y in zip(X_video_list, y_video_list):
    y_pred_raw = clf.predict(xgb.DMatrix(data=X))
    y_preds.append(y_pred_raw)
    y_pred_thresh = y_pred_raw > threshold
    class_tensor = mn.get_class_tensor(y_pred_thresh, y)
    class_tensors.append(class_tensor)

    print("Average Precision:", metrics.average_precision_score(y, y_pred_raw))
    print("AUC:", metrics.roc_auc_score(y, y_pred_raw))
    print("Confusion Matrix:", metrics.confusion_matrix(y, y_pred_thresh))

mn.visual_debugger(labeled_videos, y_preds, y_video_list, class_tensors, val_percent=0.25)