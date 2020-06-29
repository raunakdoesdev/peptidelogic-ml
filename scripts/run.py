import pickle
import os
from matplotlib.backends.backend_pdf import PdfPages

os.environ['DISPLAY'] = ':1'
import sys

sys.path.insert(0, "../")

import time
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

import plotter
from param import Param

def main():
    dfp = Param()  # default param


    if dfp.label_dlc_on:
        # input: frames to label
        # output:
        #   - dlc project
        #   - labelled frames files
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

        import mousenet as mn
        clf = pickle.load(open(dfp.path_to_clf_model, 'rb'))

        # make video paths 
        path_to_eval_videos = []
        for video_id in dfp.video_ids:
            path_to_eval_videos.append("{}/{}/{}.mp4".format(dfp.path_to_videos,video_id,video_id))

        # eval videos into dict:
        #   - key: video name
        #   - value: result value is np array in [nframes x 2]
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
            y_pred = np.load(binary_fn)[:,1]
            all_points = np.array([range(len(y_pred))]).swapaxes(0, 1)
            # dbscan = DBSCAN(eps=7, min_samples=2)
            # labels = dbscan.fit_predict(all_points, sample_weight=y_pred > 0.7)
            dbscan = DBSCAN(eps=10, min_samples=2)
            labels = dbscan.fit_predict(all_points, sample_weight=y_pred > 0.5)
            result_x, result_y = [0], [0]
            for i in range(1, len(labels)):
                if labels[i] != labels[i - 1] and labels[i] != -1:
                    result_x.append(i)
                    result_y.append(labels[i] + 1)
            result_x.append(len(labels) - 1)
            result_y.append(result_y[-1])
            result = np.array([result_x, result_y]).swapaxes(0, 1)
            result_fn = binary_fn.replace('.npy', '.cluster.npy')
            np.save(result_fn, result)


    if dfp.evaluate_matching_on: 
        
        # writes human label files  
        if dfp.eval_human_force_on:
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

        # write results 
        for video_id, matching in matchings.items():
            fn = "{}{}.match.json".format(dfp.path_to_results,video_id)
            with open(fn, 'w') as fp:
                json.dump(matching, fp, cls=NumpyEncoder, indent=2)


    if dfp.visualize_results_on:
        # input: should only use binary files and path to key files (dosage dictionaries, etc)
        # output:
        #   - compare human/machine label
        #   - precisionrecall vs training set size
        #   - dose response curve
        #   - feature ranking
        #   - temporal feature ranking
        #   - visual debugger (should be changed to accept binary file)
        #   - <...>

        if dfp.plot_compare_human_machine_instance_on:

            bkey_to_video = get_blind_key_to_video(dfp.path_to_blind_key_to_video)

            # load machine results
            machine_results = dict()
            ax_ylim = [0, 0]
            for video_id in dfp.video_ids:
                result = np.load("{}{}.npy".format(dfp.path_to_results,video_id))
                machine_results[video_id] = result

                num_events = np.sum(result[:,1])
                if num_events > ax_ylim[1]:
                    ax_ylim[1] = num_events

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

                    num_events = np.sum(result[:, 1])
                    if num_events > ax2_ylim[1]:
                        ax2_ylim[1] = num_events

            machine_cluster_results = dict()
            for video_id in dfp.video_ids:
                result = np.load('{}{}.cluster.npy'.format(dfp.path_to_results,video_id))
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


        if dfp.plot_compare_human_machine_drc_on:

            video_to_dose = get_video_to_dose(dfp)
            bkey_to_video = get_blind_key_to_video(dfp.path_to_blind_key_to_video)

            # load machine results
            machine_results = dict()
            for video_id in dfp.video_ids:
                result = np.load('{}{}.npy'.format(dfp.path_to_results,video_id))
                result = result[0:dfp.cutoff,:]
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
            for video_id in dfp.video_ids:
                result = np.load('{}{}.cluster.npy'.format(dfp.path_to_results,video_id))
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


        if dfp.plot_matching_on: 

            ax_ulim = 0

            # load match results 
            matchings = dict() 
            for video_id in dfp.video_ids: 
                fn = "{}{}.match.json".format(dfp.path_to_results,video_id)
                with open(fn, 'r') as j:
                    matchings[video_id] = json.loads(j.read())

            # load human files:
            human_results = dict()
            for video_id in dfp.video_ids: 
                fn = "{}{}.human.npy".format(dfp.path_to_results,video_id)
                human_results[video_id] = np.load(fn)

                num_events = human_results[video_id].shape[0]
                if ax_ulim < num_events:
                    ax_ulim = num_events

            # load machine files: 
            machine_results = dict()
            for video_id in dfp.video_ids: 
                fn = "{}{}.cluster.npy".format(dfp.path_to_results,video_id)
                machine_results[video_id] = np.load(fn)

                num_events = machine_results[video_id].shape[0]
                if ax_ulim < num_events:
                    ax_ulim = num_events

            # plot
            for video_id in dfp.video_ids: 
                plotter.plot_matching(machine_results[video_id], human_results[video_id], matchings[video_id],
                    [0,ax_ulim],video_id)


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
        "num_TP" : 0,
        "num_FP" : 0,
        "num_FN" : 0,
    }

    matchings = dict()

    # accounts for human delay and machine clustering only start point 
    tolerance_sec = 10
    tolerance_min = tolerance_sec / 60

    for video_id in dfp.video_ids: 

        matching = dict()
        TP = []
        FN = []
        FP = [] 
        machine_matched = []
        human_matched = []

        # get results
        human_labelled = human_results[video_id] # dim: [nevents x 2], units: [minutes x Z]
        machine_labelled = machine_results[video_id] # dim: [nevents x 2], units: [frames x Z]

        # convert 
        machine_labelled = np.asarray(machine_labelled,dtype=np.float32)
        machine_labelled[:,0] = machine_labelled[:,0] / 30.0 / 60.0 # dim: [nevents x 2], units: [minutes x Z]
        machine_labelled = machine_labelled[1:-2,:] # remove first and last 

        # make graph 
        dist = np.abs(human_labelled[:,0][:,np.newaxis] - machine_labelled[:,0])

        # two methods
        method_1_on = True
        if method_1_on:

            # solve linear program 

            dist[dist > tolerance_min] = 1000
            human_event_assignments, machine_event_assignments = linear_sum_assignment(dist) # tries to minimize this 

            for (human_event_assignment,machine_event_assignment) in zip(human_event_assignments,machine_event_assignments):
                if dist[human_event_assignment,machine_event_assignment] < tolerance_min:
                    machine_matched.append(machine_labelled[machine_event_assignment,1])
                    human_matched.append(human_labelled[human_event_assignment,1])
                    TP.append(machine_labelled[machine_event_assignment,0])

        else: 

            # 'greedy'
            if human_labelled.shape[0] > 0:

                for machine_idx, (machine_time, machine_event) in enumerate(machine_labelled): 

                    human_idx = np.argmin(dist[:,machine_idx])
                    human_time,human_event = human_labelled[human_idx,:]

                    if dist[human_idx,machine_idx] < tolerance_min and not human_event in human_matched:
                        machine_matched.append(machine_event)
                        human_matched.append(human_event)
                        TP.append(machine_time)

        for machine_time, machine_event in machine_labelled:
            if not machine_event in machine_matched:
                FP.append(machine_time)

        for human_time, human_event in human_labelled:
            if not human_event in human_matched:
                FN.append(human_time)

        stats["num_TP"] += len(TP)
        stats["num_FP"] += len(FP)
        stats["num_FN"] += len(FN)
        matching["TP"] = np.asarray(TP) 
        matching["FP"] = np.asarray(FP) 
        matching["FN"] = np.asarray(FN) 

        matchings[video_id] = matching

    stats["precision"] = stats["num_TP"] / (stats["num_TP"] + stats["num_FP"])
    stats["recall"] = stats["num_TP"] / (stats["num_TP"] + stats["num_FN"])

    print(stats)

    return stats,matchings

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self,obj)

if __name__ == '__main__':
    main()
