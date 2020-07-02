from tqdm import tqdm
from tqdm.contrib import tzip

from mousenet.vis import plotter
import seaborn as sns;
import numpy as np

sns.set()
import pandas as pd


def load(x):
    if type(x) == str and '.pkl' in x:
        return pd.read_pickle(x)
    return x


def plot_single_video_instance(human_results, machine_df, video_id):
    machine_df: pd.DataFrame = load(machine_df.format(VIDEO_ID=video_id))

    fig, ax = plotter.make_fig()
    ax2 = ax.twinx()

    # Plot lines
    human_mean = np.mean(np.array(human_results), axis=0)[:, 1].cumsum()
    human_std = np.std(np.array(human_results), axis=0)[:, 1]
    sns.lineplot(human_results[0][:, 0], y=human_mean, ax=ax, ci=None)
    ax.fill_between(human_results[0][:, 0], human_mean - human_std * 0.5, human_mean + human_std * 0.5, color='blue', alpha=0.2)
    sns.lineplot(machine_df.index.values / 60, y=machine_df['Clustered Events'].replace(-1, method='ffill') + 1, ax=ax,
                 ci=None, markevery=5000, marker='^')
    sns.lineplot(machine_df.index.values / 60, y=machine_df['Event Confidence'].cumsum(), ax=ax2,
                 ci=None, color=next(ax._get_lines.prop_cycler)['color'], markevery=5000, marker='X')

    # Formatting
    ax2.set_ylabel(r'$\sum P$')
    ax2.set_xlim([0, 30])
    ax2.grid(None)
    ax.set_xlim([0, 30])
    ax.set_ylim(bottom=0.0)
    ax.set(title=video_id, xlabel='Time (minutes)', ylabel='# of Events')
    ax.legend(ax.lines + ax2.lines, ["Human", "Machine Clustered", "Machine Raw"])


def plot_drc(human_results, machine_df_path, dose_to_videos):
    fig, ax = plotter.make_fig()
    ax2 = ax.twinx()

    for color, dose in tzip(sns.color_palette()[:len(dose_to_videos)], dose_to_videos.keys(), desc='Plotting DRC'):
        human_df, machine_df = None, None

        machine_cluster = []
        machine_raw = []
        human_cluster = [human_result[:, 1].cumsum() for human_result in human_results[dose]]
        for video_id in dose_to_videos[dose]:
            machine_df: pd.DataFrame = load(machine_df_path.format(VIDEO_ID=video_id)).head(53000)
            machine_cluster.append(machine_df['Clustered Events'].replace(-1, method='ffill') + 1)
            machine_raw.append(machine_df['Event Confidence'].cumsum())

        sns.lineplot(machine_df.index.values / 60, np.mean(np.array(machine_cluster), axis=0), marker='^', color=color,
                     ax=ax, ci=None, markevery=5000)
        sns.lineplot(machine_df.index.values / 60, np.mean(np.array(machine_raw), axis=0), marker='X', color=color,
                     ax=ax2, ci=None, markevery=5000)
        sns.lineplot(human_results[dose][0][:, 0], np.mean(np.array(human_cluster), axis=0), color=color, ax=ax,
                     ci=None)

    # Formatting
    ax2.set_ylabel(r'$\sum P$')
    ax2.set_xlim([0, 30])
    ax2.set_ylim(bottom=0.0)
    ax2.grid(None)
    ax.set_xlim([0, 30])
    ax.set_ylim(bottom=0.0)
    ax.set(title='Dose Response Curve', xlabel='Time (minutes)', ylabel='# of Events')

    for color, dose in zip(sns.color_palette()[:len(dose_to_videos)], dose_to_videos.keys()):
        ax2.plot(np.nan, np.nan, color=color, label=dose)
    handles, labels = ax2.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: int(t[0].split(' ')[0])))
    ax2.legend(handles, labels, loc='upper left')

    for marker, type in zip(['^', 'X', None], ['Machine Clustered', 'Machine Raw', 'Human']):
        ax.plot(np.nan, np.nan, marker=marker, label=type, color='black', markeredgecolor='white', markeredgewidth=1)
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles)))
    ax.legend(handles, labels, loc='center left')


def get_colors(some_dict):
    colors = dict()
    fig, ax = plt.subplots()
    for key, value in some_dict.items():
        line = ax.plot(np.nan, np.nan)
        colors[key] = line[0].get_color()
    plt.close(fig)
    return colors
