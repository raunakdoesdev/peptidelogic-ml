import os

import PySimpleGUI as sg
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import mousenet as mn
import pandas as pd
import logging

sg.theme('LightGrey1')  # Define GUI Theme


class VideoBrowser:
    def __init__(self, videos, scaling=0.6):
        self.scaling = scaling
        self.videos = videos
        self.video_paths = [video.path for video in videos]
        self.video = videos[0]
        self.frame_num = self.video.start
        self.play = False
        self.video_window = [[sg.Combo(values=self.video_paths, default_value=self.video.path, readonly=True,
                                       key='Switch Vids', size=(int(105 * self.scaling), int(10 * self.scaling)),
                                       enable_events=True)],
                             [sg.Image(key='video')]]
        self.controls_window = [[sg.Slider(range=(self.video.start, self.video.end), default_value=self.video.start,
                                           size=(int(70 * self.scaling), int(10 * self.scaling)),
                                           orientation='h', key='Frame Slider', enable_events=True,
                                           disable_number_display=True),
                                 sg.Spin([str(i) for i in range(self.video.start, self.video.end)],
                                         size=(int(9 * self.scaling), int(10 * self.scaling)), key='Frame Picker',
                                         initial_value=str(self.video.start)),
                                 sg.Button('GO', size=(int(5 * self.scaling), 1)),
                                 sg.Combo(values=[-10, -5, -3, -1, 1, 3, 5, 10], default_value=1, key='Speed',
                                          readonly=True)]]

    def get_windows(self):
        return self.video_window, self.controls_window

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

    def update_controls(self, window, values, event):
        frame_num = self.frame_num
        if event == 'p':
            self.play = not self.play
        if event == 'Frame Slider':
            frame_num = int(values['Frame Slider'])
            window['Frame Picker'].update(str(frame_num))
        elif event == 'GO':
            frame_num = int(values['Frame Picker'])
            window['Frame Slider'].update(frame_num)
        elif self.play or event == 'd':
            frame_num += float(values['Speed'])
            window['Frame Slider'].update(frame_num)
            window['Frame Picker'].update(str(frame_num))
        elif self.play or event == 'a':
            frame_num -= float(values['Speed'])
            window['Frame Picker'].update(str(frame_num))
            window['Frame Slider'].update(frame_num)
        elif event == 'r':
            window['Speed'].update(-float(values['Speed']))
        elif event == 'Switch Vids':
            print("HELLO")
            self.video = self.videos[self.video_paths.index(values['Switch Vids'])]
            self.frame_num = self.video.start
            window['Frame Slider'].update(self.video.start, range=(self.video.start, self.video.end))
            window['Frame Picker'].update(self.frame_num)
        else:
            return False

        if self.video.start <= frame_num <= self.video.end:
            self.frame_num = frame_num
            return True
        else:
            return False


class ErrorBrowser:
    def __init__(self, error_list, scaling=1.0):
        self.browser = [[sg.Listbox(values=[], size=(None, 15))]]
        pass

    def get_windows(self):
        return self.browser

    def update(self, window, values, event):
        pass
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


class VisualDebugger:
    def __init__(self, video: mn.LabeledVideo, y, y_hat, scaling=0.8):
        self.scaling = scaling
        sg.theme('LightGrey1')  # Define GUI Theme
        import pickle
        # self.read2time, self.frame2read = pickle.load(open('maps.pkl', 'rb'))

        # # Scaled window
        # layout = [[sg.Text('')]]
        # scaling_window = sg.Window('Window Title', layout, no_titlebar=True, auto_close=False,
        #                            alpha_channel=0).Finalize()
        # scaling_window.TKroot.tk.call('tk', 'scaling', self.scaling)
        # scaling_window.close()

        self.y = y
        self.y_hat = y_hat
        self.dpts = (y, y_hat)

        # Define frame parameters
        self.cap = cv2.VideoCapture(video.path)  # .labeled_video_path)
        self.start_frame = video.start
        self.end_frame = video.end
        self.frame_num = self.start_frame
        self.video = video

        self.df = pd.read_hdf(video.df_path)
        self.df = self.df[self.df.columns.get_level_values(0).unique()[0]]

        # print(f"FPS: {self.cap.get(cv2.CAP_PROP_FPS)}")

        self.play = False

        layout = [[sg.Text("     ", background_color='white'), sg.Image(key='video')],
                  [sg.Canvas(key='canvas')],
                  [sg.Text("     ", background_color='white'),
                   sg.Slider(range=(self.start_frame, self.end_frame), default_value=self.start_frame,
                             size=(int(55 * self.scaling), int(10 * self.scaling)),
                             orientation='h', key='Frame Slider', enable_events=True, disable_number_display=True),
                   sg.Spin([str(i) for i in range(self.start_frame, self.end_frame)],
                           size=(int(9 * self.scaling), int(10 * self.scaling)), key='Frame Picker',
                           initial_value=str(self.start_frame)),
                   sg.Button('GO', size=(int(5 * self.scaling), 1)),
                   sg.Combo(values=[-10, -5, -3, -1, 1, 3, 5, 10], default_value=1, key='Speed', readonly=True)]]

        self.window = sg.Window('Visual Debugger', layout, return_keyboard_events=True)

        self.window.read(timeout=0)
        self._init_plot()
        self._update_video()
        while self._event_loop(): pass
        self.window.close()

    def _event_loop(self):
        event, values = self.window.read(timeout=0)
        if event in (None, 'Exit'):
            return False

        if self._update_state(event, values):  # if something changed
            self._update_plot()
            self._update_video()
        return True

    def _init_plot(self):
        self.fig = Figure(figsize=(9 * self.scaling, 4 * self.scaling))
        self.axes = []
        for axis in range(1, len(self.dpts) + 1):
            self.axes.append(self.fig.add_subplot(len(self.dpts), 1, axis))
            self.axes[-1].grid()
            self.axes[-1].xaxis.set_visible(False)

        self.fig.tight_layout()
        self.fig_agg = self._draw_figure(self.window['canvas'].TKCanvas)

    def _update_state(self, event, values):
        frame_num = self.frame_num
        if event == 'p':
            self.play = not self.play
        if event == 'Frame Slider':
            frame_num = int(values['Frame Slider'])
            self.window['Frame Picker'].update(str(frame_num))
        elif event == 'GO':
            frame_num = int(values['Frame Picker'])
            self.window['Frame Slider'].update(frame_num)
            print("Hello")
        elif self.play or event == 'd':
            frame_num += float(values['Speed'])
            self.window['Frame Slider'].update(frame_num)
            self.window['Frame Picker'].update(str(frame_num))
        elif self.play or event == 'a':
            frame_num -= float(values['Speed'])
            self.window['Frame Picker'].update(str(frame_num))
            self.window['Frame Slider'].update(frame_num)
        elif event == 'r':
            self.window['Speed'].update(-float(values['Speed']))
        else:
            return False

        if self.start_frame < frame_num < self.end_frame:
            self.frame_num = frame_num

        return True

    def _update_video(self):
        keys = (('hindpaw_right', (0, 255, 0)), ('hindpaw_left', (0, 255, 0)), ('hindheel_right', (0, 0, 255)),
                ('hindheel_left', (0, 255, 0)), ('frontpaw_left', (0, 255, 0)), ('frontpaw_right', (0, 255, 0)),
                ('tail', (0, 255, 0)),)

        try:
            if 'CFS' in self.video.path:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.frame_num))
            else:
                self.cap.set(cv2.CAP_PROP_POS_MSEC, self.video.win_map[int(self.frame_num)])

            _, frame = self.cap.read()
            for key, color in keys:
                if self.df[key]['likelihood'][self.frame_num] > 0.9:
                    x, y = self.df[key]['x'][self.frame_num], self.df[key]['y'][self.frame_num]
                    cv2.circle(frame, (int(x), int(y)), 3, color, 6)
            frame = cv2.resize(frame, (int(frame.shape[1] * self.scaling), int(frame.shape[0] * self.scaling)),
                               interpolation=cv2.INTER_AREA)
            imgbytes = cv2.imencode('.png', frame)[1].tobytes()
            self.window['video'].update(data=imgbytes)
        except Exception as e:
            logging.exception(e)

    def _draw_figure(self, canvas):
        figure_canvas_agg = FigureCanvasTkAgg(self.fig, canvas)
        figure_canvas_agg.draw()
        figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
        return figure_canvas_agg

    def _update_plot(self):
        try:
            for axis, data in zip(self.axes, self.dpts):
                axis.cla()
                axis.grid()
                axis.set_ylim([-0.2, 1.2])
                axis.axvline(0, c='r')
                idx = int(self.frame_num - self.start_frame)
                axis.plot(range(100), data[idx: idx + 100])
            self.fig_agg.draw()
        except Exception as e:
            logging.exception(e)
