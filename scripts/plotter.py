import matplotlib.pyplot as plt
import numpy as np
import os, subprocess
from matplotlib.backends.backend_pdf import PdfPages

# default param 
plt.rcParams.update({'font.size': 15})
plt.rcParams['lines.linewidth'] = 2.5  # 2.5


def save_figs(filename):
    fn = os.path.join(os.getcwd(), filename)
    pp = PdfPages(fn)
    for i in plt.get_fignums():
        plt.figure(i).tight_layout()
        pp.savefig(plt.figure(i))
        plt.close(plt.figure(i))
    pp.close()


def open_figs(filename):
    pdf_path = os.path.join(os.getcwd(), filename)
    if os.path.exists(pdf_path):
        subprocess.call(["xdg-open", pdf_path])


def make_fig():
    fig, ax = plt.subplots()
    ax.grid(True)
    return fig, ax


def show():
    plt.show()


def plot_instance_over_time_machine(machine_result, fig, ax, color, ax_ylim):

    times = machine_result[:, 0]
    events = machine_result[:, 1]

    cum_events = np.cumsum(events)

    ax.plot(times, cum_events, color=color)
    ax.set_ylabel(r'$\sum P$')
    ax.set_xlabel('minutes')
    ax.set_ylim(ax_ylim)


def plot_instance_over_time_machine_cluster(machine_result, fig, ax, color, ax_ylim):
    from scipy.interpolate import interp1d
    interp = interp1d([0, 50000], [0, 30])

    times = interp(machine_result[:, 0])
    events = machine_result[:, 1]

    ax2 = ax.twinx()
    ax2.plot(times, events, color=color, marker='x', label='machine cluster')
    ax2.set_ylim(ax_ylim)


def plot_instance_over_time_human(human_result, fig, ax, color, ax_ylim):
    human_result = np.asarray(human_result)

    times = human_result[0][:, 0]
    mean_events = np.mean(human_result[:, :, 1], axis=0)
    std_events = np.std(human_result[:, :, 1], axis=0)
    cum_events = np.cumsum(mean_events)

    ax2 = ax.twinx()
    ax2.plot(times, cum_events, color=color, marker='o')
    ax2.fill_between(times,
                     cum_events - std_events,
                     cum_events + std_events, alpha=0.5, color=color)

    ax2.plot(np.nan, np.nan, color=color, marker='o', label='human')
    ax2.plot(np.nan, np.nan, color=color, marker='x', label='machine-cluster')
    ax2.plot(np.nan, np.nan, color=color, label='machine')

    ax2.set_ylabel('# events')
    ax2.set_xticks(times)
    ax2.legend(loc='upper left')
    ax2.set_ylim(ax_ylim)


def plot_drc_machine(machine_results, fig, ax, colors):
    for dose, machine_result in machine_results.items():
        machine_result = np.asarray(machine_result)  # shape = [n cases in drc, n frames, 2]
        times = machine_result[0][:, 0]
        mean_events = np.mean(machine_result[:, :, 1], axis=0)
        std_events = np.std(machine_result[:, :, 1], axis=0)

        cum_events = np.cumsum(mean_events)

        ax.plot(times, cum_events, color=colors[dose], label=dose)
        ax.fill_between(times,
                        cum_events - std_events,
                        cum_events + std_events, alpha=0.5, color=colors[dose])
        ax.set_ylabel(r'$\sum P$')
        ax.set_xlabel('minutes')


def plot_drc_machine_cluster(machine_results, fig, ax, colors):
    ax2 = ax.twinx()
    from scipy.interpolate import interp1d
    times = list(range(0, 31, 3))
    interp = interp1d([0, 50000 - 1], [0, 30])
    backinterp = interp1d([0, 30], [0, 50000 - 1])

    for dose, machine_result in machine_results.items():
        dose_results = []
        for vid in machine_result:
            vid_result = np.interp(backinterp(times), vid[:, 0], vid[:, 1])
            print(vid_result)
            dose_results.append(vid_result)
        dose_results = np.array(dose_results)
        mean_events = np.mean(dose_results, axis=0)
        std_events = np.std(dose_results, axis=0)

        ax2.plot(times, mean_events, color=colors[dose], marker='x', label=dose)
        # ax2.fill_between(times,
        #                  mean_events - std_events,
        #                  mean_events + std_events, alpha=0.5, color=colors[dose])
    ax2.set_ylim([0, 100])



def plot_drc_human(human_results, fig, ax, colors):
    ax2 = ax.twinx()

    for dose, human_result in human_results.items():
        human_result = np.asarray(human_result)
        times = human_result[0][:, 0]
        mean_events = np.mean(human_result[:, :, 1], axis=0)
        std_events = np.std(human_result[:, :, 1], axis=0)
        cum_events = np.cumsum(mean_events)

        ax2.plot(times, cum_events, color=colors[dose], marker='o')
        ax2.fill_between(times,
                         cum_events - std_events,
                         cum_events + std_events, alpha=0.1, color=colors[dose])

    for dose, color in colors.items():
        ax2.plot(np.nan, np.nan, color=color, label=dose)

    handles, labels = ax2.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: int(t[0].split(' ')[0])))
    ax2.legend(handles, labels, loc='upper left')
    ax2.set_ylabel('# events')
    ax2.set_xticks(times)
    ax2.set_ylim([0, 100])


def plot_matchings(machine_results, human_results, matchings):

    video_ids = machine_results.keys()

    for video_id in video_ids: 

        machine_result = machine_results[video_id]
        human_result = human_results[video_id]
        matching = matchings[video_id]

        # convert
        # machine_result[:,0] = machine_result[:,0] #/ 60.0 # to minutes 
        machine_result[:,0] = machine_result[:,0] / 60 / 30 # to minutes 

        fig,ax = plt.subplots()
        ax.set_title(video_id)
        ax.plot(machine_result[:,0],machine_result[:,1],marker='o',label='machine')
        ax.plot(human_result[:,0],human_result[:,1],marker='s',label='human')

        for machine_event_match, human_event_match in matching.items(): 
            
            machine_time_match = machine_result[machine_result[:,1] == machine_event_match,0][0]
            human_time_match = human_result[human_result[:,1] == human_event_match,0][0]

            ax.plot(\
                [machine_time_match,human_time_match],\
                [machine_event_match,human_event_match],color='green',alpha=0.5)

        ax.plot(np.nan,np.nan,color='green',alpha=0.5,label='match')
        ax.grid(True)
        ax.legend()




def get_colors(some_dict):
    colors = dict()
    fig, ax = plt.subplots()
    for key, value in some_dict.items():
        line = ax.plot(np.nan, np.nan)
        colors[key] = line[0].get_color()
    plt.close(fig)
    return colors
