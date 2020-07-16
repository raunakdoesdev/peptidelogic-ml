from tqdm import tqdm
from tqdm.contrib import tzip

from mousenet.vis import plotter
import seaborn as sns
import numpy as np
import pandas as pd
import os

if 'DISPLAY' not in os.environ:  # Use TeamViewer if SSH'd
    os.environ['DISPLAY'] = ':1'


def load(x):
    if type(x) == str and '.pkl' in x:
        return pd.read_pickle(x)
    return x


def plot_single_video_instance(human_results, machine_df, video_id, matching_stats, plot_raw=False, y_top=None):
    machine_df: pd.DataFrame = load(machine_df.format(VIDEO_ID=video_id))

    fig, ax = plotter.make_fig()

    # Plot lines
    for human_mean in human_results:
        sns.lineplot(human_results[0][:, 0], y=human_mean, ax=ax, ci=None)
    sns.lineplot(machine_df.index.values / 60, y=machine_df['Clustered Events'].replace(-1, method='ffill') + 1, ax=ax,
                 ci=None, markevery=5000, marker='^')

    if plot_raw:
        ax2 = ax.twinx()
        sns.lineplot(machine_df.index.values / 60, y=machine_df['Event Confidence'].cumsum(), ax=ax2,
                     ci=None, color=next(ax._get_lines.prop_cycler)['color'], markevery=5000, marker='X')

        ax2.set_ylabel(r'$\sum P$')
        ax2.set_xlim([0, 30])
        ax2.grid(None)
        ax.legend(ax.lines + ax2.lines, ["Human", "Machine Clustered", "Machine Raw"])
    else:
        ax.legend(ax.lines, ["Human", "Machine Clustered"])

    ax.set_xlim([0, 30])
    ax.set_ylim(bottom=0.0, top=y_top)
    ax.set(title=video_id, xlabel='Time (minutes)', ylabel='# of Events')

    if matching_stats is not None:
        alpha = 0.3
        for tp in matching_stats['TP']:
            ax.axvline(x=tp, color='green', alpha=alpha)
        for fp in matching_stats['FP']:
            ax.axvline(x=fp, color='red', alpha=alpha)
        for fn in matching_stats['FN']:
            ax.axvline(x=fn, color='orange', alpha=alpha)


def prauc_vs_data(X_train, X_test, y_train, y_test):
    import xgboost, mousenet, sklearn
    train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999]

    prauc = []
    for train_size in tqdm(train_sizes, desc='PRAUC vs. Data'):
        X_train_new, _, y_train_new, _ = mousenet.train_test_split(X_train, y_train, train_size=train_size,
                                                                   shuffle=True)
        clf = mousenet.train_xgboost(X_train_new, y_train_new, X_test, y_test, verbose_eval=False)
        prauc.append(sklearn.metrics.average_precision_score(y_test, clf.predict(xgboost.DMatrix(data=X_test),
                                                                                 ntree_limit=clf.best_ntree_limit)))

    fig, ax = plotter.make_fig()
    ax.plot([s * 100 for s in train_sizes], prauc)
    ax.set_ylabel('PRAUC on Fixed Validation Set')
    ax.set_xlabel('Training Data (%)')
    ax.set_ylim([0, 1])


def plot_drc(human_results, machine_df_path, dose_to_videos, plot_raw=True):
    fig, ax = plotter.make_fig()
    ax2 = ax.twinx()

    for color, dose in tzip(sns.color_palette()[:len(dose_to_videos)], dose_to_videos.keys(), desc='Plotting DRC'):
        human_df, machine_df = None, None

        machine_cluster = []
        machine_raw = []
        human_cluster = [human_result[:, 1].cumsum() for human_result in human_results[dose]]
        for video_id in dose_to_videos[dose]:
            video_id = 'CFR' + video_id
            machine_df: pd.DataFrame = load(machine_df_path.format(VIDEO_ID=video_id)).head(53000)
            machine_cluster.append(machine_df['Clustered Events'].replace(-1, method='ffill') + 1)
            machine_raw.append(machine_df['Event Confidence'].cumsum())

        sns.lineplot(machine_df.index.values / 60, np.mean(np.array(machine_cluster), axis=0), marker='^', color=color,
                     ax=ax, ci=None, markevery=5000)
        if plot_raw:
            sns.lineplot(machine_df.index.values / 60, np.mean(np.array(machine_raw), axis=0), marker='X', color=color,
                         ax=ax2, ci=None, markevery=5000)
        sns.lineplot(human_results[dose][0][:, 0], np.mean(np.array(human_cluster), axis=0), color=color, ax=ax,
                     ci=None)

    # Formatting
    if plot_raw:
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

    if plot_raw:
        for marker, type in zip(['^', 'X', None], ['Machine Clustered', 'Machine Raw', 'Human']):
            ax.plot(np.nan, np.nan, marker=marker, label=type, color='black', markeredgecolor='white',
                    markeredgewidth=1)
    else:
        for marker, type in zip(['^', None], ['Machine Clustered', 'Human']):
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
