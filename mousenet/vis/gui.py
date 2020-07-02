import logging
import os

import PySimpleGUI as sg
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import mousenet as mn

sg.theme('LightGrey1')  # Define GUI Theme


def visual_debugger(videos, y_preds, y, class_tensors, val_percent=None, scaling=1.0):
    if 'DISPLAY' not in os.environ:  # Use TeamViewer if SSH'd
        os.environ['DISPLAY'] = ':1'

    video_paths = [video.path for video in videos]
    video = videos[0]

    video_switcher = sg.Combo(video_paths, default_value=video_paths[0], readonly=True,
                              key='Switch Vids', size=(int(80 * scaling), int(10 * scaling)),
                              enable_events=True)

    vb = VideoBrowser(video, scaling=0.5, control_scaling=0.8)
    eb = ErrorBrowser(error_list=class_tensors[0], scaling=1)
    gp = GUIPlots(dpts=[y[0], y_preds[0]], scaling=scaling, val_percent=val_percent)

    # Get Windows
    video_window, control_window = vb.get_windows()
    event_window = eb.get_windows()
    plot_window = gp.get_windows()

    layout = [[video_switcher], [video_window, event_window], [control_window], [plot_window]]

    window = sg.Window('Visual Debugger', layout, return_keyboard_events=True)
    window.read(timeout=0)
    vb.update_frame(window)
    gp.init_plot(window)
    while True:
        event, values = window.read(timeout=0)
        if event in (None, 'Exit'):
            break
        elif event == 'Switch Vids':
            vid_idx = video_paths.index(values['Switch Vids'])
            vb.video = videos[vid_idx]
            vb.frame_num = vb.video.start
            window['Frame Slider'].update(vb.video.start, range=(vb.video.start, vb.video.end))
            window['Frame Picker'].update(int(vb.frame_num),
                                          values=[str(i) for i in range(vb.video.start, vb.video.end)])
            window['event_list'].update(eb.process_error_list(class_tensors[vid_idx]))
            gp.dpts = [y[vid_idx], y_preds[vid_idx]]

        update = eb.update(window, values, event)
        if update:
            vb.set_frame_num_offset_start(window, update)
        vb.update(window, values, event)
        gp.update_plot(vb.frame_num - vb.video.start)
    window.close()


class VideoBrowser:
    def __init__(self, video, scaling=0.6, control_scaling=1.0):
        self.scaling = scaling
        self.video = video
        self.frame_num = self.video.start
        self.play = False
        self.video_window = [[sg.Image(key='video')]]
        self.controls_window = [[sg.Slider(range=(self.video.start, self.video.end), default_value=self.video.start,
                                           size=(int(70 * control_scaling), int(10 * control_scaling)),
                                           orientation='h', key='Frame Slider', enable_events=True,
                                           disable_number_display=True),
                                 sg.Spin([str(i) for i in range(self.video.start, self.video.end)],
                                         size=(int(9 * control_scaling), int(10 * control_scaling)), key='Frame Picker',
                                         initial_value=str(self.video.start)),
                                 sg.Button('GO', size=(int(5 * control_scaling), 1)),
                                 sg.Combo(values=[-10, -5, -3, -1, 1, 3, 5, 10], default_value=1, key='Speed',
                                          readonly=True)]]

    def get_windows(self):
        return sg.Column(self.video_window), sg.Column(self.controls_window)

    def update_frame(self, window):
        try:
            frame = self.video.grab_frame_with_bparts(self.frame_num)
            frame = cv2.resize(frame, (int(frame.shape[1] * self.scaling), int(frame.shape[0] * self.scaling)),
                               interpolation=cv2.INTER_AREA)
            imgbytes = cv2.imencode('.png', frame)[1].tobytes()
            window['video'].update(data=imgbytes)
        except Exception as e:
            logging.exception(e)

    def update(self, window, values, event):
        change = self.update_controls(window, values, event)
        if change:
            self.update_frame(window)
        return change

    def set_frame_num_offset_start(self, window, frame_num):
        frame_num += self.video.start
        self.frame_num = frame_num
        window['Frame Picker'].update(str(int(frame_num - self.video.start)))
        window['Frame Slider'].update(frame_num)
        self.update_frame(window)

    def update_controls(self, window, values, event):
        frame_num = self.frame_num
        if event == 'p':
            self.play = not self.play
        elif event == 'r':
            window['Speed'].update(-float(values['Speed']))
        if event == 'Frame Slider':
            frame_num = int(values['Frame Slider'])
            window['Frame Picker'].update(str(int(frame_num - self.video.start)))
        elif event == 'GO':
            frame_num = int(float(values['Frame Picker']) + self.video.start)
            window['Frame Slider'].update(frame_num)
        elif self.play or event == 'd':
            frame_num += float(values['Speed'])
            window['Frame Slider'].update(frame_num)
            window['Frame Picker'].update(str(int(frame_num - self.video.start)))
        elif event == 'a':
            frame_num -= float(values['Speed'])
            window['Frame Picker'].update(str(int(frame_num - self.video.start)))
            window['Frame Slider'].update(frame_num)
        else:
            return True

        if self.video.start <= frame_num <= self.video.end:
            self.frame_num = frame_num
            return True
        else:
            return True


class ErrorBrowser:
    def __init__(self, error_list, scaling=1.0):
        self.browser = [[sg.Listbox(values=self.process_error_list(error_list), size=(26, 16), enable_events=True,
                                    key='event_list')]]
        pass

    def process_error_list(self, error_list):
        self.event_changes = np.where(error_list[:-1] != error_list[1:])[0]
        self.event_changes = [idx + 1 for idx in self.event_changes]
        self.event_change_names = []
        for idx in self.event_changes:
            self.event_change_names.append(mn.ClassificationTypes(error_list[idx]).name)
        self.error_list = [f'{idx} - {event}' for idx, event in zip(self.event_changes, self.event_change_names)]
        return self.error_list

    def get_windows(self):
        return sg.Column(self.browser)

    def update(self, window, values, event):
        if event == 'event_list':
            return int(self.event_changes[self.error_list.index(values['event_list'][0])])
        # change = self.update_controls(window, values, event)
        # if change:
        #     self.update_frame(window)
        # return change


if __name__ == '__main__':
    os.environ['DISPLAY'] = ':1'
    dlc = mn.DLCProject(config_path='/home/pl/Retraining-BenR-2020-05-25/config.yaml')
    vids = mn.folder_to_videos('/home/pl/Data/mWT SR 017 (PL 100960 DRC IV)_renamed', labeled=True,
                               required_words=('CFS',))
    dlc.infer_trajectories(vids, infer=False)
    vb = VideoBrowser(vids)
    eb = ErrorBrowser(error_list=None)
    w1, w2 = vb.get_windows()
    w3 = eb.get_windows()
    layout = [[sg.Column([[sg.Column(w1)], [sg.Column(w2)]]), sg.Column(w3)]]
    window = sg.Window('Visual Debugger', layout, return_keyboard_events=True)
    window.read(timeout=0)
    vb.update_frame(window)
    while True:
        event, values = window.read(timeout=0)
        if event in (None, 'Exit'):
            break
        vb.update(window, values, event)
        eb.update(window, values, event)
    window.close()


class GUIPlots:
    def __init__(self, dpts, val_percent=None, scaling=1.0):
        self.plots = [[sg.Canvas(key='canvas')]]
        self.val_percent = val_percent
        self.scaling = scaling
        self.dpts = dpts

    def init_plot(self, window):
        self.fig = Figure(figsize=(7 * self.scaling, 2 * self.scaling))
        self.axes = []
        for axis in range(1, len(self.dpts) + 1):
            self.axes.append(self.fig.add_subplot(len(self.dpts), 1, axis))
            self.axes[-1].grid()
            self.axes[-1].xaxis.set_visible(False)

        self.fig.tight_layout()
        self.fig_agg = self._draw_figure(window['canvas'].TKCanvas)

    def _draw_figure(self, canvas):
        figure_canvas_agg = FigureCanvasTkAgg(self.fig, canvas)
        figure_canvas_agg.draw()
        figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
        return figure_canvas_agg

    def update_plot(self, idx):
        idx = int(idx)
        try:
            for axis, data in zip(self.axes, self.dpts):
                axis.cla()
                axis.grid()
                axis.set_ylim([-0.2, 1.2])
                axis.axvline(0, c='r')
                if 100 + idx < (1 - self.val_percent) * len(data):
                    rect = plt.Rectangle((0., 0.), 100, 1, fill=True, alpha=0.2)
                    axis.add_patch(rect)
                elif idx < (1 - self.val_percent) * len(data):
                    rect = plt.Rectangle((0., 0.), ((1 - self.val_percent) * len(data)) - idx, 1, fill=True,
                                         alpha=0.2)
                    axis.add_patch(rect)
                axis.plot(range(100), data[idx: idx + 100])
            self.fig_agg.draw()
        except Exception as e:
            logging.exception(e)

    def get_windows(self):
        return sg.Column(self.plots)