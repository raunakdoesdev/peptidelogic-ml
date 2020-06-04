from sklearn import ensemble
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import mousenet as mn
import torch
from sklearn.utils import resample
import os
import xgboost as xgb


dlc = mn.DLCProject(config_path='/home/pl/Retraining-BenR-2020-05-25/config.yaml')
labeled_videos = mn.json_to_videos('/home/pl/Data', '../benv2-synced.json', mult=1)
dlc.infer_trajectories(labeled_videos)

video = labeled_videos[0]
import pandas as pd

dfs = []
ys = []
for video in labeled_videos:
    if 'Writhe' not in video.ground_truth:
        continue
    df = pd.read_hdf(video.df_path)
    df = df[df.columns.get_level_values(0).unique()[0]]
    df = df.iloc[int(video.start): int(video.end)]

    # cols = []
    # for col in df.columns:
    #     if 'likelihood' in col[1]:
    #         cols.append(col)
    # df = df[cols]

    dfs.append(df)

    y = torch.load(video.ground_truth['Writhe']).numpy()
    ys.append(y)

df = pd.concat(dfs, ignore_index=True)
y = np.concatenate(ys)
X = df[df.columns]

# window_size = 20
# new_df = df.copy(deep=True)
# for i in range(-window_size // 2, 1 + window_size // 2):
#     for col in df.columns:
#         new_df[col[0], col[1] + str(i)] = df[col].shift(i)
#
# new_df.fillna(value=0, inplace=True)
# X = new_df[new_df.columns]

split = 0.25
X_test = X[:int(split * len(X))]
X_train = X[int(split * len(X)):]
y_test = y[:int(split * len(y))]
y_train = y[int(split * len(y)):]

print(X_test.shape)
print(X_train.shape)
print(y_test.shape)
print(y_train.shape)

data_dmatrix = xgb.DMatrix(data=X,label=y)


clf = ensemble.AdaBoostClassifier()

# Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

from sklearn import metrics

print("Average Precision:", metrics.average_precision_score(y_test, y_pred))
