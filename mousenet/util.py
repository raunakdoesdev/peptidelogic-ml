from itertools import islice
import logging
import torch


def in_range(event_ranges, frame_num):
    return any([int(range_min) <= int(frame_num) < int(range_max) for range_min, range_max in event_ranges])


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
