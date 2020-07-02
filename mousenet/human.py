import os

import numpy as np
import pandas as pd
import json


def extract_human_labels(key_to_video_id, human_label_files, save_path, force=False):
    """
    Reads data from human label files and saves in similar format to machine labels.

    :param key_to_video_id: dictionary mapping blinded video keys to video ids
    :param human_label_files: list of human label excel files
    :return:
    """

    blind_key_to_mouse_times = dict()

    for human_label_file in human_label_files:
        for human_label_file in human_label_files:
            bsr_xls = pd.ExcelFile(human_label_file)

            chamber_data = pd.read_excel(bsr_xls, bsr_xls.sheet_names[0]).to_numpy()[5:, 1]
            chamber_to_blind_key = {chamber: blind_key for chamber, blind_key in enumerate(chamber_data)}

            chamber_to_event_times = dict()

            chamber_row_start = 26
            chamber_col_idxs = 1 + 4 * np.arange(0, 6)
            bsr_pd = pd.read_excel(bsr_xls, bsr_xls.sheet_names[1])
            bsr_np = bsr_pd.to_numpy()

            for chamber, chamber_col_idx in enumerate(chamber_col_idxs):
                chamber_data = bsr_np[chamber_row_start:, chamber_col_idx]
                chamber_data = chamber_data[~pd.isnull(chamber_data)]
                chamber_to_event_times[chamber] = chamber_data

            for chamber, event_times in chamber_to_event_times.items():
                blind_key_to_mouse_times[chamber_to_blind_key[chamber]] = chamber_to_event_times[chamber]

    for blind_key, mouse_times in blind_key_to_mouse_times.items():
        mouse_times = np.asarray(mouse_times, dtype=np.float32)
        df = pd.DataFrame(data={'Event': 1 + np.arange(0, mouse_times.shape[0])}, index=mouse_times)
        df.to_pickle(save_path.format(VIDEO_ID=key_to_video_id[str(blind_key)]))


def get_video_to_dose(path_to_video_to_dose):
    # video to dose
    video_to_dose = dict()
    video_to_dose_pd = pd.read_excel(path_to_video_to_dose)
    for row in range(video_to_dose_pd.shape[0]):
        video_id = video_to_dose_pd.iat[row, 0]
        dose = video_to_dose_pd.iat[row, 1]
        video_to_dose[video_id] = dose
    return video_to_dose


def extract_drc_human_labels(video_ids, human_label_file, key_to_video_id_file, video_to_dose_file):
    # load human results
    human_results = dict()
    bsr_xls = pd.ExcelFile(human_label_file)
    key_to_video_id = json.load(open(key_to_video_id_file))
    video_to_dose = get_video_to_dose(video_to_dose_file)

    for sheet_name in bsr_xls.sheet_names:
        bsr_pd = pd.read_excel(bsr_xls, sheet_name)
        bsr_np = bsr_pd.to_numpy()
        times = bsr_pd.columns.to_numpy()[1:-1]

        for row in bsr_np:
            bkey = str(row[0])
            video_id = key_to_video_id[bkey]

            if video_id in video_ids:

                events = row[1:-1]
                dose = video_to_dose[video_id]

                result = np.zeros((times.shape[0], 2))
                result[:, 0] = times
                result[:, 1] = events

                if dose in human_results.keys():
                    human_results[dose].append(result)
                else:
                    human_results[dose] = [result]

    return human_results

def extract_video_human_labels(video_ids, human_label_file, key_to_video_id_file, video_to_dose_file):
    # load human results
    human_results = dict()
    bsr_xls = pd.ExcelFile(human_label_file)
    key_to_video_id = json.load(open(key_to_video_id_file))
    video_to_dose = get_video_to_dose(video_to_dose_file)

    for sheet_name in bsr_xls.sheet_names:
        bsr_pd = pd.read_excel(bsr_xls, sheet_name)
        bsr_np = bsr_pd.to_numpy()
        times = bsr_pd.columns.to_numpy()[1:-1]

        for row in bsr_np:
            bkey = str(row[0])
            video_id = key_to_video_id[bkey]

            if video_id in video_ids:

                events = row[1:-1]
                dose = video_to_dose[video_id]

                result = np.zeros((times.shape[0], 2))
                result[:, 0] = times
                result[:, 1] = events
                if dose in human_results.keys():
                    human_results[video_id].append(result)
                else:
                    human_results[video_id] = [result]

    return human_results
