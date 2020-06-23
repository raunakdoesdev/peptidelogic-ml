import pickle
import os
from matplotlib.backends.backend_pdf import PdfPages

import sys

sys.path.insert(0, "../")

import time
import numpy as np
import pandas as pd
import plotter
import json
from param import Param
from tqdm import tqdm


def main():
    dfp = Param()  # default param

    if dfp.label_dlc_on:
        # input: frames to label
        # output:
        # 	- dlc project
        # 	- labelled frames files
        # todo

        # actually this might not need the input to frames to label, because its just calling that one gui
        # it might be good to print location of the folder

        pass

    if dfp.retrain_dlc_on:
        # input: path to labelled frames files
        # output: trained model (written to file)
        # todo
        pass

    if dfp.label_clf_on:
        # input: path to video file (needs to be constant frame rate!!)
        # output: labelled behavior frames (written to file)
        # todo

        # actually this might not need the input path to video file, because its just calling that one gui

        # maybe print this, or, put it in gui
        if not 'CFR' in dfp.path_to_clf_label_video:
            print('CLASSIFIER LABELING VIDEO NEEDS TO BE CONSTANT FRAME RATE!!')
            exit()

        pass

    if dfp.retrain_clf_on:
        # input: path to labelled behavior frames file
        # output: trained model (written to file)
        # todo
        pass

    if dfp.evaluate_vid_on:
        # input: lst of paths to video file (does not have to be constant frame rate)
        # output: times, prediction value (written to file)

        # TODO...
        # 	- implement force commands
        # 	- make results a 2d array
        # 	- debug... why doesnt running a 'blind' mouse work? i.e.
        # 	- (?) downsample 'result' from 30fps to 5fps

        import mousenet as mn
        clf = pickle.load(open(dfp.path_to_clf_model, 'rb'))

        # eval videos into dict:
        # 	- key: video name
        #	- value: result value is np array in [nframes x 2]
        result_by_video = dict()
        for video_path in tqdm(dfp.path_to_eval_videos):
            result = mn.evaluate_video(video_path, clf, dfp.path_to_dlc_project, dfp.clf_force, dfp.dlc_force)
            result_by_video[video_path] = result

        # write results to binary files
        for video_path, result in result_by_video.items():
            video_id = os.path.basename(video_path).split('.mp4')[0]
            result_fn = '{}{}.npy'.format(dfp.path_to_results, video_id)
            np.save(result_fn, result)

    if dfp.visualize_results_on:
        # input: should only use binary files and path to key files (dosage dictionaries, etc)
        # output:
        # 	- compare human/machine label
        # 	- precisionrecall vs training set size
        # 	- dose response curve
        # 	- feature ranking
        # 	- temporal feature ranking
        # 	- visual debugger (should be changed to accept binary file)
        # 	- <...>

        if dfp.compare_human_machine_instance:

            bkey_to_video = get_blind_key_to_video(dfp.path_to_bkey_to_video)

            # load machine results
            machine_results = dict()
            ax_ylim = [0, 0]
            for binary_fn in dfp.path_to_vis_files:
                video_id = os.path.basename(binary_fn).split('.npy')[0]
                result = np.load(binary_fn)
                machine_results[video_id] = result
                if np.sum(result) > ax_ylim[1]:
                    ax_ylim[1] = sum(np.load(binary_fn))

            # load human results
            human_results = dict()
            ax2_ylim = [0, 0]
            bsr_xls = pd.ExcelFile(dfp.path_to_blind_summary_report)
            for sheet_name in bsr_xls.sheet_names:
                bsr_pd = pd.read_excel(bsr_xls, sheet_name)
                bsr_np = bsr_pd.to_numpy()
                times = bsr_pd.columns.to_numpy()[1:-1]

                for row in bsr_np:
                    bkey = str(row[0])
                    video_id = bkey_to_video[bkey]
                    events = row[1:-1]

                    result = np.zeros((times.shape[0], 2))
                    result[:, 0] = times
                    result[:, 1] = events

                    if video_id in human_results.keys():
                        human_results[video_id].append(result)
                    else:
                        human_results[video_id] = [result]

                    if np.sum(result[:, 1]) > ax2_ylim[1]:
                        ax2_ylim[1] = np.sum(result[:, 1])

            # plot
            colors = plotter.get_colors(machine_results)
            for video_id, machine_result in machine_results.items():
                if video_id in human_results.keys():
                    fig, ax = plotter.make_fig()
                    ax.set_title(video_id)
                    plotter.plot_instance_over_time_machine(machine_result, fig, ax, colors[video_id], ax_ylim)
                    plotter.plot_instance_over_time_human(human_results[video_id], fig, ax, colors[video_id], ax2_ylim)

        if dfp.compare_human_machine_drc:

            video_to_dose = get_video_to_dose(dfp)
            bkey_to_video = get_blind_key_to_video(dfp.path_to_bkey_to_video)

            # load machine results
            machine_results = dict()
            for binary_fn in dfp.path_to_vis_files:
                video_id = os.path.basename(binary_fn).split('.npy')[0]
                result = np.load(binary_fn)
                dose = video_to_dose[video_id]

                if dose in machine_results.keys():
                    machine_results[dose].append(result)
                else:
                    machine_results[dose] = [result]

            # load human results
            human_results = dict()
            bsr_xls = pd.ExcelFile(dfp.path_to_blind_summary_report)
            for sheet_name in bsr_xls.sheet_names:
                bsr_pd = pd.read_excel(bsr_xls, sheet_name)
                bsr_np = bsr_pd.to_numpy()
                times = bsr_pd.columns.to_numpy()[1:-1]

                for row in bsr_np:
                    bkey = str(row[0])
                    video_id = bkey_to_video[bkey]

                    # to only use data in 'path_to_vis_files'
                    if True:  # video_id in machine_results.keys():

                        events = row[1:-1]
                        dose = video_to_dose[video_id]

                        result = np.zeros((times.shape[0], 2))
                        result[:, 0] = times
                        result[:, 1] = events

                        if dose in human_results.keys():
                            human_results[dose].append(result)
                        else:
                            human_results[dose] = [result]

                        #
            colors = plotter.get_colors(machine_results)
            fig, ax = plotter.make_fig()
            ax.set_title('Dose Response Curve')
            plotter.plot_drc_machine(machine_results, fig, ax, colors)
            plotter.plot_drc_human(human_results, fig, ax, colors)

        plotter.save_figs(dfp.plot_fn)
        plotter.open_figs(dfp.plot_fn)


def get_video_to_dose(param):
    # video to dose
    video_to_dose = dict()
    video_to_dose_pd = pd.read_excel(param.path_to_video_to_dose)
    for row in range(video_to_dose_pd.shape[0]):

        temp = True
        if temp:
            blind_key_to_video = get_blind_key_to_video(param.path_to_bkey_to_video)
            blind_key = video_to_dose_pd.iat[row, 0]
            video_id = blind_key_to_video[str(blind_key)]

        else:
            video_id = video_to_dose_pd.iat[row, 0]

        dose = video_to_dose_pd.iat[row, 1]
        video_to_dose[video_id] = dose

    return video_to_dose


def get_blind_key_to_video(path_to_bkey_to_video):
    bkey_to_video = dict()
    with open(path_to_bkey_to_video, 'r') as j:
        bkey_to_video = json.loads(j.read())
    return bkey_to_video


if __name__ == '__main__':
    main()
