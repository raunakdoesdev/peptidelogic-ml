from itertools import islice
import logging
import torch
import numpy as np


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
