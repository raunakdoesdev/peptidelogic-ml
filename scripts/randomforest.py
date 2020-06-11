import os
import pickle
from enum import Enum

import numpy as np
from sklearn import ensemble
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm
import multiprocessing as mp
import mousenet as mn
import xgboost as xgb

dlc = mn.DLCProject(config_path='/home/pl/Retraining-BenR-2020-05-25/config.yaml')
labeled_videos = mn.json_to_videos('/home/pl/Data', '../2020-06-03_ben-synced.json', mult=1)
dlc.infer_trajectories(labeled_videos)
X_train, X_test, y_train, y_test, X, y = mn.get_sklearn_dataset(labeled_videos, window_size=100, test_size=0.25)

# X_train, X_test, y_train, y_test = train_test_split(labeled_videos, window_size=20, test_size=0.25)

# train_test_split(X, y, test_size=0.1, shuffle=False)

# data_dmatrix = xgb.DMatrix(data=X, label=y)
#
# cv_results = xgb.cv(dtrain=data_dmatrix, params={'objective': 'binary:logistic', 'nthread': mp.cpu_count(), },
#                     nfold=3, shuffle=False, num_boost_round=50, early_stopping_rounds=10, metrics="aucpr",
#                     as_pandas=True, seed=123)

data_dmatrix = xgb.DMatrix(data=X_train, label=y_train)
val_dmatrix = xgb.DMatrix(data=X_test, label=y_test)


def prauc(pred, dtrain):
    lab = dtrain.get_label()
    return 'PRAUC', metrics.average_precision_score(lab, pred)


model_path = 'XGB.save'
force = False

if force or not os.path.exists(model_path):
    clf = xgb.train(dtrain=data_dmatrix, params={'objective': 'binary:logistic', 'nthread': mp.cpu_count(), },
                    num_boost_round=200, evals=[(val_dmatrix, 'val_set')], early_stopping_rounds=30,
                    feval=prauc, maximize=True)
    pickle.dump(clf, open(model_path, 'wb'))
else:
    clf = pickle.load(open(model_path, 'rb'))

y_pred_raw = clf.predict(xgb.DMatrix(data=X_test))
print('Average Precision', metrics.average_precision_score(y_test, y_pred_raw))
print("AUC", metrics.roc_auc_score(y_test, y_pred_raw))

# kf = KFold(n_splits=4)
# metrics_log = mn.util.AverageLogger()
# for train_index, test_index in tqdm(kf.split(X)):
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#     # clf = ensemble.RandomForestClassifier(n_estimators=300, verbose=True, n_jobs=multiprocessing.cpu_count(), bootstrap=False)
#     clf = xgboost.XGBClassifier()
#     clf.fit(X_train, y_train)
#
#     y_pred_raw = clf.predict(X_test)
#     metrics_log.update('Average Precision', metrics.average_precision_score(y_test, y_pred_raw))
#     metrics_log.update("AUC", metrics.roc_auc_score(y_test, y_pred_raw))
#     metrics_log.update("Confusion Matrix", metrics.confusion_matrix(y_test, y_pred_raw))
#
# metrics_log.print()

#
# # #
# # # # Infer on Full Video --> Visualize
threshold = 0.7
y_pred_raw = clf.predict(xgb.DMatrix(data=X))
y_pred_thresh = y_pred_raw > threshold


class ClassificationTypes(Enum):
    TRUE_NEGATIVE = 0
    TRUE_POSITIVE = 1
    FALSE_NEGATIVE = 2
    FALSE_POSITIVE = 3


class_tensor = ((y_pred_thresh == 0) * (y == 0)) * ClassificationTypes.TRUE_NEGATIVE.value + \
               ((y_pred_thresh == 1) * (y == 1)) * ClassificationTypes.TRUE_POSITIVE.value + \
               ((y_pred_thresh == 0) * (y == 1)) * ClassificationTypes.FALSE_NEGATIVE.value + \
               ((y_pred_thresh == 1) * (y == 0)) * ClassificationTypes.FALSE_POSITIVE.value

print("Average Precision:", metrics.average_precision_score(y, y_pred_raw))
print("AUC:", metrics.roc_auc_score(y, y_pred_raw))
print("Confusion Matrix:", metrics.confusion_matrix(y, y_pred_thresh))
print(class_tensor)

#
