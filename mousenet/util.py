from enum import Enum
from itertools import islice
import logging
import torch
import numpy as np
from sklearn import metrics


def in_range(event_ranges, frame_num, map=None):
    if map is None:
        return any([int(min) <= int(frame_num) < int(max) for min, max in event_ranges])
    if map is not None:
        return any([int(map(min)) <= int(frame_num) < int(map(max)) for min, max in event_ranges])


class DisableLogger():
    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, a, b, c):
        logging.disable(logging.NOTSET)


def dist(df, bp1, bp2):
    return (df[bp1]['x'] * df[bp2]['x']) ** 2 + (df[bp1]['y'] * df[bp2]['y']) ** 2


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 - 1)
    result[0::2] = lst
    return result


class View(torch.nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def __repr__(self):
        return f'View{self.args}'

    def forward(self, x):
        return x.view(*self.args)


class Max(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def __repr__(self):
        return f'Max{self.dim}'

    def forward(self, x):
        return torch.max(x, dim=self.dim)[0]


def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)


class AverageLogger:
    def __init__(self):
        self.dic = {}

    def update(self, key, new_val):
        if key in self.dic:
            self.dic[key].append(new_val)
        else:
            self.dic[key] = [new_val]

    def print(self):
        for key in self.dic.keys():
            print(f'{key} -> {np.mean(self.dic[key])}')


class ClassificationTypes(Enum):
    TRUE_NEGATIVE = 0
    TRUE_POSITIVE = 1
    FALSE_NEGATIVE = 2
    FALSE_POSITIVE = 3


def get_class_tensor(y_pred_thresh, y):
    class_tensor = ((y_pred_thresh == 0) * (y == 0)) * ClassificationTypes.TRUE_NEGATIVE.value + \
                   ((y_pred_thresh == 1) * (y == 1)) * ClassificationTypes.TRUE_POSITIVE.value + \
                   ((y_pred_thresh == 0) * (y == 1)) * ClassificationTypes.FALSE_NEGATIVE.value + \
                   ((y_pred_thresh == 1) * (y == 0)) * ClassificationTypes.FALSE_POSITIVE.value
    return class_tensor


def prauc_feval(pred, dtrain):
    lab = dtrain.get_label()
    return 'PRAUC', metrics.average_precision_score(lab, pred)


# Data Frame Helpers
class BodypartProcessor:
    def __init__(self, df):
        self.df = df

    def area(self, a, b, c):
        return (self.df[a]['x'] * (self.df[b]['y'] - self.df[c]['y']) +
                self.df[b]['x'] * (self.df[c]['y'] - self.df[a]['y']) +
                self.df[c]['x'] * (self.df[a]['y'] - self.df[b]['y'])).abs() / 2

    def distance(self, bp1, bp2):
        return ((self.df[bp1]['x'] - self.df[bp2]['x']) ** 2 + \
               (self.df[bp1]['y'] - self.df[bp2]['y']) ** 2) ** (1/2)

    def middle(self, a, b):
        self.df[f'{a}-{b}-middle', 'x'] = (self.df[a]['x'] + self.df[b]['x']) / 2
        self.df[f'{a}-{b}-middle', 'y'] = (self.df[a]['y'] + self.df[b]['y']) / 2
        return f'{a}-{b}-middle'

    def prune(self):
        cols = []
        for col in self.df.columns:
            if 'likelihood' in col[1] or 'feature' in col[1]:
                cols.append(col)
        return self.df[cols]

    def __setitem__(self, key, value):
        self.df[key, 'feature'] = value
