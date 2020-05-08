import PySimpleGUI as sg
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import mousenet as mn


class VisualDebugger:
    def __init__(self, video: mn.LabeledVideo, div=1):
        # Define frame parameters
        self.start_frame = int(video.start / div)
        self.end_frame = int(video.end / div)
        self.cur_frame = self.start_frame

        cap = cv2.VideoCapture(video.labeled_video_path)

        num_frames = len(dpts[0])
        end_frame = video.end

        layout = [[sg.Text("     ", background_color='white'), sg.Image(key='video')],
                  [sg.Canvas(key='canvas')],
                  [sg.Text("     ", background_color='white'),
                   sg.Slider(range=(self.start_frame, self.end_frame), default_value=self.start_frame, size=(55, 10),
                             orientation='h', key='Frame Slider', enable_events=True, disable_number_display=True),
                   sg.Spin([str(i) for i in range(int(num_frames) - 2)], size=(9, 10), key='Frame Picker',
                           initial_value=str(self.start_frame)),
                   sg.Button('GO', size=(5, 1))], ]

        sg.theme('LightGrey1')
        window = sg.Window('Test', layout, return_keyboard_events=True)
        event, values = window.read(timeout=0)
        video_state = _init_plot(video_state, window, dpts)
        while True:
            event, values = window.read(timeout=0)
            video_state = _update_state(event, window, values, video_state)
            _update_plot(video_state, dpts)
            _update_video(cap, window, values['Frame Slider'])

    def _init_plot(state, window, dpts):
        num_graphs = 2
        state['fig'] = Figure(figsize=(9, 4))
        for axis in range(1, num_graphs + 1):
            state[f'ax{axis}'] = state['fig'].add_subplot(num_graphs, 1, axis)
            state[f'ax{axis}'].grid()
            state[f'ax{axis}'].xaxis.set_visible(False)
        state['fig'].tight_layout()
        state['fig_agg'] = _draw_figure(window['canvas'].TKCanvas, state['fig'])
        return state


def _update_state(event, window, values, state):
    frame_num = state.get('frame_num', 0)
    play = state.get('play', False)

    if event == 'p':
        play = not play
    if event == 'Frame Slider':
        frame_num = int(values['Frame Slider'])
        window['Frame Picker'].update(str(frame_num))
    elif event == 'GO':
        frame_num = int(values['Frame Picker'])
        window['Frame Slider'].update(frame_num)
    elif play:
        window['Frame Slider'].update(frame_num)
        window['Frame Picker'].update(str(frame_num))

    state['play'] = play
    frame_num = frame_num + 4 if play else frame_num

    if state['start_frame'] < frame_num < state['end_frame']:
        state['frame_num'] = frame_num

    return state


def _update_video(cap, window, frame_num):
    # Plot Video
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        _, frame = cap.read()
        imgbytes = cv2.imencode('.png', frame)[1].tobytes()
        window['video'].update(data=imgbytes)
    except:
        pass


def _draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def _update_plot(state, dpts):
    num_graphs = 2
    for axis in range(1, num_graphs + 1):
        state[f'ax{axis}'].cla()
        state[f'ax{axis}'].grid()
        state[f'ax{axis}'].set_ylim([-0.2, 1.2])
        state[f'ax{axis}'].axvline(0, c='r')
        state[f'ax{axis}'].plot(range(100), dpts[axis - 1][state['frame_num']:state['frame_num'] + 100].detach())
    state['fig_agg'].draw()


if __name__ == '__main__':
    main()
