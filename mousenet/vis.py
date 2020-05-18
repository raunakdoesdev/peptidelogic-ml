import os

if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = 'localhost:10.0'  # Enable X11 Forwarding

import PySimpleGUI as sg
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import mousenet as mn
import pandas as pd
import logging

if __name__ == '__main__':
    layout = [[sg.Text('My one-shot window.')],
              [sg.InputText()],
              [sg.Submit(), sg.Cancel()]]

    window = sg.Window('Window Title', layout)

    event, values = window.read()
    window.close()

    text_input = values[0]
    sg.popup('You entered', text_input)


class AlignmentChecker:

    def __init__(self, video: mn.LabeledVideo):
        sg.theme('LightGrey1')  # Define GUI Theme
        import json
        self.time2frame = json.load(open('time2frame.json', 'r'))

        # Define frame parameters
        self.video = video
        self.df = pd.read_hdf(video.df_path)
        self.df = self.df[self.df.columns.get_level_values(0).unique()[0]]
        self.cap = cv2.VideoCapture(video.path, cv2.CAP_FFMPEG)
        self.num_frames = int(self.video.get_num_frames())
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.start_frame = 5
        self.end_frame = self.num_frames
        self.frame_num = self.start_frame
        self.cur_frame = 0
        self.play = False

        layout = [[sg.Image(key='video')],
                  [sg.Slider(range=(self.start_frame, self.end_frame), default_value=self.start_frame,
                             size=(55, 10),
                             orientation='h', key='Frame Slider', enable_events=True, disable_number_display=True),
                   sg.Spin([str(i) for i in range(self.start_frame, self.end_frame)], size=(9, 10),
                           key='Frame Picker',
                           initial_value=str(self.start_frame)),
                   sg.Button('GO', size=(5, 1)),
                   sg.Combo(values=[-10, -5, -3, -1, 1, 3, 5, 10], default_value=1, key='Speed', readonly=True)],
                  [sg.Slider(range=(90800, 100000), default_value=1, size=(55, 10), orientation='h', key='Offset',
                             enable_events=True)]]

        self.window = sg.Window('Alignment Checker', layout, return_keyboard_events=True)

        self.first = True
        self.window.read(timeout=0)
        while self._event_loop(): pass
        self.window.close()

    def _event_loop(self):
        event, values = self.window.read(timeout=0)
        if event in (None, 'Exit'):
            return False

        if self.first or self._update_state(event, values):  # if something changed
            self._update_video(values)
            self.first = False
        return True

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
        elif self.play or event == 'd':
            self.window['Frame Slider'].update(frame_num)
            self.window['Frame Picker'].update(str(frame_num))
            frame_num += float(values['Speed'])
        elif self.play or event == 'a':
            self.window['Frame Slider'].update(frame_num)
            self.window['Frame Picker'].update(str(frame_num))
            frame_num -= float(values['Speed'])
        elif event == 'r':
            self.window['Speed'].update(-float(values['Speed']))
        elif event == 'Offset':
            return True
        else:
            return False

        if self.start_frame < frame_num < self.end_frame:
            self.frame_num = frame_num

        return True

    def _update_video(self, values):
        keys = (('leftear', (0, 255, 0)), ('rightear', (0, 255, 0)), ('tail', (0, 0, 255)))
        try:
            print(self.fps)
            # self.cap.set(cv2.CAP_PROP_POS_FRAMES, int((self.frame_num * self.fps/30) - 1))
            self.cap.set(cv2.CAP_PROP_POS_MSEC, self.time2frame[str(int(self.frame_num))])
            _, frame = self.cap.read()
            for key, color in keys:
                if self.df[key]['likelihood'][self.frame_num] > 0.9:
                    x, y = self.df[key]['x'][self.frame_num], self.df[key]['y'][self.frame_num]
                    cv2.circle(frame, (int(x), int(y)), 3, color, 6)
            imgbytes = cv2.imencode('.png', frame)[1].tobytes()
            self.window['video'].update(data=imgbytes)
        except Exception as e:
            logging.exception(e)


class VisualDebugger:
    def __init__(self, video: mn.LabeledVideo, y, y_hat):
        sg.theme('LightGrey1')  # Define GUI Theme
        import pickle
        # self.read2time, self.frame2read = pickle.load(open('maps.pkl', 'rb'))

        self.y = y
        self.y_hat = y_hat
        self.dpts = (y, y_hat)

        # Define frame parameters
        self.cap = cv2.VideoCapture(video.path)  # .labeled_video_path)
        self.start_frame = video.orig_start
        self.end_frame = video.orig_end
        self.frame_num = self.start_frame
        self.video = video

        self.df = pd.read_hdf(video.df_path)
        self.df = self.df[self.df.columns.get_level_values(0).unique()[0]]

        # print(f"FPS: {self.cap.get(cv2.CAP_PROP_FPS)}")

        self.play = False

        layout = [[sg.Text("     ", background_color='white'), sg.Image(key='video')],
                  [sg.Canvas(key='canvas')],
                  [sg.Text("     ", background_color='white'),
                   sg.Slider(range=(self.start_frame, self.end_frame), default_value=self.start_frame, size=(55, 10),
                             orientation='h', key='Frame Slider', enable_events=True, disable_number_display=True),
                   sg.Spin([str(i) for i in range(self.start_frame, self.end_frame)], size=(9, 10), key='Frame Picker',
                           initial_value=str(self.start_frame)),
                   sg.Button('GO', size=(5, 1)),
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
        self.fig = Figure(figsize=(9, 4))
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
        elif self.play or event == 'd':
            self.window['Frame Slider'].update(frame_num)
            self.window['Frame Picker'].update(str(frame_num))
            frame_num += float(values['Speed'])
        elif self.play or event == 'a':
            frame_num -= float(values['Speed'])
        elif event == 'r':
            self.window['Speed'].update(-float(values['Speed']))
        else:
            return False

        if self.start_frame < frame_num < self.end_frame:
            self.frame_num = frame_num

        return True

    def _update_video(self):
        keys = (('leftear', (0, 255, 0)), ('rightear', (0, 255, 0)), ('tail', (0, 0, 255)))

        try:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, self.video.read2time[int(self.frame_num)])
            _, frame = self.cap.read()
            for key, color in keys:
                if self.df[key]['likelihood'][self.frame_num] > 0.9:
                    x, y = self.df[key]['x'][self.frame_num], self.df[key]['y'][self.frame_num]
                    cv2.circle(frame, (int(x), int(y)), 3, color, 6)
            imgbytes = cv2.imencode('.png', frame)[1].tobytes()
            self.window['video'].update(data=imgbytes)
        except Exception:
            pass

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
                axis.plot(range(100), data[idx: idx + 100].detach())
            self.fig_agg.draw()
        except Exception as e:
            logging.exception(e)
