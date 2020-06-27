import pickle
import os
from matplotlib.backends.backend_pdf import PdfPages

os.environ['DISPLAY'] = ':1'
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
        # 	- debug... why doesnt running a 'blind' mouse work? i.e.
        # 	- (?) downsample 'result' from 30fps to 5fps
        #   - make runtime efficient

        import mousenet as mn
        clf = pickle.load(open(dfp.path_to_clf_model, 'rb'))

        # make video paths 
        path_to_eval_videos = []
        for video_id in dfp.video_ids:
            path_to_eval_videos.append("{}/{}/{}.mp4".format(dfp.path_to_videos,video_id,video_id))

        # eval videos into dict:
        # 	- key: video name
        #	- value: result value is np array in [nframes x 2]
        result_by_video = dict()
        for video_path in tqdm(path_to_eval_videos):
            result = mn.evaluate_video(video_path, clf, dfp.path_to_dlc_project, dfp.clf_force, dfp.dlc_force)
            result_by_video[video_path] = result

        # write results to binary files
        for video_path, result in result_by_video.items():
            video_id = os.path.basename(video_path).split('.mp4')[0]
            result_fn = '{}{}.npy'.format(dfp.path_to_results, video_id)
            np.save(result_fn, result)


    if dfp.evaluate_cluster_on:

        # make results files paths 
        path_to_vis_files = []
        for video_id in dfp.video_ids:
            path_to_vis_files.append("{}{}.npy".format(dfp.path_to_results,video_id))

        from sklearn.cluster import DBSCAN
        for binary_fn in tqdm(path_to_vis_files, desc="Computing Clusters"):
            # y_pred = np.load(binary_fn)
            # all_points = np.array([range(len(y_pred))]).swapaxes(0, 1)
            # dbscan = DBSCAN(eps=7, min_samples=2)
            # dbscan = DBSCAN(eps=1, min_samples=2)
            # labels = dbscan.fit_predict(all_points, sample_weight=y_pred > 0.7)
            # result_x, result_y = [0], [0]
            # for i in range(1, len(labels)):
            #     if labels[i] != labels[i - 1] and labels[i] != -1:
            #         result_x.append(i)
            #         result_y.append(labels[i] + 1)
            # result_x.append(len(labels) - 1)
            # result_y.append(result_y[-1])

            result = np.load(binary_fn)
            y_pred = result[:,1]
            all_points = np.array([result[:,0]]).swapaxes(0, 1)

            dbscan = DBSCAN(eps=1, min_samples=2)
            labels = dbscan.fit_predict(all_points, sample_weight=y_pred > 0.7)
            result_x, result_y = [0], [0]
            for i in range(1, len(labels)):
                if labels[i] != labels[i - 1] and labels[i] != -1:
                    # result_x.append(all_points[i])
                    result_x.append(i)
                    result_y.append(labels[i] + 1)
            # result_x.append(all_points[-1])
            result_x.append(len(labels) - 1)
            result_y.append(result_y[-1])

            result = np.array([result_x, result_y]).swapaxes(0, 1)
            result = np.array(result,dtype=np.float32)
            result_fn = binary_fn.replace('.npy', '.cluster.npy')
            np.save(result_fn, result)


    if dfp.evaluate_matching_on: 
        
        # writes human label files  
        evaluate_human_labels(dfp, dfp.paths_to_human_label_files) 

        # load human files:
        human_results = dict()
        for video_id in dfp.video_ids: 
            fn = "{}{}.human.npy".format(dfp.path_to_results,video_id)
            human_results[video_id] = np.load(fn)

        # load machine files: 
        machine_results = dict()
        for video_id in dfp.video_ids: 
            fn = "{}{}.cluster.npy".format(dfp.path_to_results,video_id)
            machine_results[video_id] = np.load(fn)

        stats, matchings = matching_stats(dfp,machine_results,human_results)

        plotter.plot_matchings(machine_results,human_results,matchings)
        

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

        # make paths
        path_to_vis_files = []
        for video_id in dfp.video_ids:
            path_to_vis_files.append("{}{}.npy".format(dfp.path_to_results,video_id))


        if dfp.compare_human_machine_instance:

            bkey_to_video = get_blind_key_to_video(dfp.path_to_blind_key_to_video)

            # load machine results
            machine_results = dict()
            ax_ylim = [0, 0]
            for binary_fn in path_to_vis_files:
                video_id = os.path.basename(binary_fn).split('.npy')[0]
                result = np.load(binary_fn)
                machine_results[video_id] = result

                if np.sum(result[:,1]) > ax_ylim[1]:
                    ax_ylim[1] = np.sum(result[:,1])

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

            machine_cluster_results = dict()
            for binary_fn in path_to_vis_files:
                video_id = os.path.basename(binary_fn).split('.npy')[0]
                result = np.load(binary_fn.replace('.npy', '.cluster.npy'))
                machine_cluster_results[video_id] = result
                if result[-1][1] > ax2_ylim[1]:
                    ax2_ylim[1] = result[-1][1]

            # plot
            colors = plotter.get_colors(machine_results)
            for video_id, machine_result in machine_results.items():
                if video_id in human_results.keys():
                    fig, ax = plotter.make_fig()
                    ax.set_title(video_id)
                    plotter.plot_instance_over_time_machine(machine_result, fig, ax, colors[video_id], ax_ylim)
                    plotter.plot_instance_over_time_machine_cluster(machine_cluster_results[video_id], fig, ax,
                                                                    colors[video_id], ax2_ylim)
                    plotter.plot_instance_over_time_human(human_results[video_id], fig, ax, colors[video_id], ax2_ylim)


        if dfp.compare_human_machine_drc:

            video_to_dose = get_video_to_dose(dfp)
            bkey_to_video = get_blind_key_to_video(dfp.path_to_blind_key_to_video)

            # load machine results
            machine_results = dict()
            for binary_fn in path_to_vis_files:
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

                    if video_id in dfp.video_ids: 

                        events = row[1:-1]
                        dose = video_to_dose[video_id]

                        result = np.zeros((times.shape[0], 2))
                        result[:, 0] = times
                        result[:, 1] = events

                        if dose in human_results.keys():
                            human_results[dose].append(result)
                        else:
                            human_results[dose] = [result]

            machine_cluster_results = dict()
            for binary_fn in path_to_vis_files:
                video_id = os.path.basename(binary_fn).split('.npy')[0]
                result = np.load(binary_fn.replace('.npy', '.cluster.npy'))
                dose = video_to_dose[video_id]

                if dose in machine_cluster_results.keys():
                    machine_cluster_results[dose].append(result)
                else:
                    machine_cluster_results[dose] = [result]

            colors = plotter.get_colors(machine_results)
            fig, ax = plotter.make_fig()
            ax.set_title('Dose Response Curve')
            plotter.plot_drc_machine(machine_results, fig, ax, colors)

            plotter.plot_drc_machine_cluster(machine_cluster_results, fig, ax, colors)
            plotter.plot_drc_human(human_results, fig, ax, colors)

        plotter.save_figs(dfp.plot_fn)
        plotter.open_figs(dfp.plot_fn)


def get_video_to_dose(param):
    # video to dose
    video_to_dose = dict()
    video_to_dose_pd = pd.read_excel(param.path_to_video_to_dose)
    for row in range(video_to_dose_pd.shape[0]):
        video_id = video_to_dose_pd.iat[row, 0]
        dose = video_to_dose_pd.iat[row, 1]
        video_to_dose[video_id] = dose
    return video_to_dose


def get_blind_key_to_video(path_to_blind_key_to_video):
    blind_key_to_video = dict()
    with open(path_to_blind_key_to_video, 'r') as j:
        blind_key_to_video = json.loads(j.read())
    return blind_key_to_video


def evaluate_human_labels(dfp, human_label_files):

    # takes data from human_label_files
    # writes data into binary files like cluster data

    for human_label_file in human_label_files:

        # first get blind key to video id 
        blind_key_to_video_id = get_blind_key_to_video(dfp.path_to_blind_key_to_video)

        # next get blind key to mouse times 
        blind_key_to_mouse_times = dict()

        for human_label_file in human_label_files: 

            bsr_xls = pd.ExcelFile(human_label_file)

            # in first sheet, grab mouse chamber to blind key 
            chamber_to_blind_key = dict() 

            row_start = 5
            col = 1 

            bsr_pd = pd.read_excel(bsr_xls, bsr_xls.sheet_names[0])
            bsr_np = bsr_pd.to_numpy()
            blind_keys = bsr_np[row_start:,col] 

            for chamber, blind_key in enumerate(blind_keys):
                chamber_to_blind_key[chamber] = blind_key 


            # in next sheet, grab mouse event times 
            chamber_to_event_times = dict()

            chamber_row_start = 26
            chamber_col_idxs = 1 + 4 * np.arange(0,6) 

            bsr_pd = pd.read_excel(bsr_xls, bsr_xls.sheet_names[1])
            bsr_np = bsr_pd.to_numpy()

            for chamber, chamber_col_idx in enumerate(chamber_col_idxs):

                chamber_data = bsr_np[chamber_row_start:,chamber_col_idx]
                chamber_data = chamber_data[~pd.isnull(chamber_data)]
                chamber_to_event_times[chamber] = chamber_data


            # assign 
            for chamber, event_times in chamber_to_event_times.items():
                blind_key_to_mouse_times[chamber_to_blind_key[chamber]] = chamber_to_event_times[chamber]

    # write files 
    for blind_key, mouse_times in blind_key_to_mouse_times.items():
        video_id = blind_key_to_video_id[str(blind_key)]
        human_result_filename = '{}{}.human.npy'.format(dfp.path_to_results,video_id)

        mouse_times = np.asarray(mouse_times,dtype=np.float32)
        n_events = mouse_times.shape[0]
        events = 1 + np.arange(0,n_events)

        human_result = np.zeros((n_events,2))
        human_result[:,0] = mouse_times
        human_result[:,1] = events

        np.save(human_result_filename, human_result)    


def matching_stats(dfp,machine_results,human_results):

    stats = {
        'num_TP': 0,
        'num_FP': 0,
        'num_FN': 0,
    }

    matchings = dict()

    # accounts for human delay and machine clustering only start point 
    tolerance_sec = 30
    tolerance_min = tolerance_sec / 60

    for video_id in dfp.video_ids: 

        matching = dict()

        # get results
        human_labelled = human_results[video_id] # dim: [nevents x 2], units: [minutes x Z]
        machine_labelled = machine_results[video_id] # dim: [nevents x 2], units: [frames x Z]

        # convert 
        machine_labelled = np.asarray(machine_labelled,dtype=np.float32)
        # machine_labelled[:,0] = machine_labelled[:,0] / 60.0 # dim: [nevents x 2], units: [minutes x Z]
        machine_labelled[:,0] = machine_labelled[:,0] / 30.0 / 60.0 # dim: [nevents x 2], units: [minutes x Z]

        # make graph 
        dist = np.abs(human_labelled[:,0][:,np.newaxis] - machine_labelled[:,0])

        # two methods
        method_1_on = True
        if method_1_on:

            # solve linear program 

            from scipy.optimize import linear_sum_assignment

            dist[dist > tolerance_min] = 1000
            human_event_assignments, machine_event_assignments = linear_sum_assignment(dist) # tries to minimize this 

            for (human_event_assignment,machine_event_assignment) in zip(human_event_assignments,machine_event_assignments):
                if dist[human_event_assignment,machine_event_assignment] < tolerance_min:
                    matching[machine_labelled[machine_event_assignment,1]] = human_labelled[human_event_assignment,1]
                    stats["num_TP"] += 1 

        else: 

            # 'greedy'

            for machine_idx, (machine_time, machine_event) in enumerate(machine_labelled): 
                for human_idx, (human_time, human_event) in enumerate(human_labelled):
                    if dist[human_idx,machine_idx] < tolerance_min and not human_event in matching.values():
                        matching[machine_event] = human_event
                        stats["num_TP"] += 1                     


        not_matched_machine = []
        for machine_event in machine_labelled[:,1]:
            if not machine_event in matching.keys():
                not_matched_machine.append(machine_event)
        stats["num_FP"] += len(not_matched_machine)


        not_matched_human = []
        for human_event in human_labelled[:,1]:
            if not human_event in matching.values():
                not_matched_human.append(human_event)
        stats["num_FN"] += len(not_matched_human)


        matchings[video_id] = matching


    stats["precision"] = stats["num_TP"] / (stats["num_TP"] + stats["num_FP"])
    stats["recall"] = stats["num_TP"] / (stats["num_TP"] + stats["num_FN"])

    print(stats)

    return stats,matchings



if __name__ == '__main__':
    main()
