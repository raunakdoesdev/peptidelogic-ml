import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.absolute().parent.absolute()))
import PySimpleGUI as sg
import os
import cv2
import json
import numpy as np
from tqdm import tqdm
import mousenet as mn
import pickle

import logging

def launcher():
    layout = [
        [sg.Listbox([], size=(65, 3), key='video_list')],
        [sg.Button('Add Videos'), sg.Button('Clear Videos')],
        [sg.Button('Launch Labeler')]]
    window = sg.Window('Enter Frame Number', layout)
    current = 0
    while True:
        event, values = window.read()
        if event == 'Add Videos':
            window['video_list'].update(window['video_list'].GetListValues() + list(sg.filedialog.askopenfilenames()))
        elif event == 'Clear Videos':
            window['video_list'].update([])
        elif event == 'Launch Labeler':
            try:
                videos = window['video_list'].GetListValues()
                if all([os.path.exists(video) and video.endswith('.mp4') for video in videos]):
                    window.close()
                    return videos
                else:
                    raise Exception('Videos in Incorrect Format. Please Clear and Try Again.')
            except Exception as e:
                sg.popup_error(f'Invalid Values Provided. Specific Error:\n{e}')
        if event is None or event == 'Exit':
            window.close()
            raise Exception('Exited without launching labeler.')


def good_focus(window):
    return True


def labeled(frame_num):
    return False


def video_labeler(video_paths):
    videos = [mn.LabeledVideo(video) for video in video_paths]
    dlc = mn.DLCProject(config_path='/home/pl/Retraining-BenR-2020-05-25/config.yaml')
    dlc.infer_trajectories(videos)

    video = videos[0]

    video_dic = {}

    label_dash = [[sg.Text('Event Class: '),
                   sg.Combo(['Itch', 'Writhe', 'WallStand', 'NoMove'], size=(10, 1), default_value="Itch",
                            key='EventClass')],
                  [sg.Text('Labeler: ', size=(5, 1)), sg.Input(size=(15, 1), key='Labeler')],
                  [sg.Button('Load Labels', size=(9, 1)), sg.Button('Save Labels', size=(9, 1))],
                  [sg.Text("Shortcuts:\n"
                           "A - Subtract {speed} frames\n"
                           "D - Add {speed} frames\n"
                           "R - Negate speed (reverse direction)\n"
                           "P - Play/Pause\n"
                           "L - Start/End Label")]]

    video_dash = [[sg.Text('Video Selector:'),
                   sg.Combo(video_paths, default_value=video_paths[0], readonly=True, size=(105, 3), enable_events=True,
                            key='Video Selector')],
                  [sg.Image(filename='', key='image')],
                  [sg.Slider(range=(0, video.get_num_frames()), size=(65, 10), orientation='h', key='Frame Slider',
                             enable_events=True, disable_number_display=True),
                   sg.Spin([str(i) for i in range(video.get_num_frames())], size=(9, 10), key='Frame Picker'),
                   sg.Button('GO', size=(4, 1))],
                  [sg.Button('Play', key='PlayPause', size=(8, 1)), sg.Text('Play Speed:'),
                   sg.Combo([-5, -3, -2, -1.5, -1.25, -1, -0.5, -0.25, 0.25, 0.5, 1, 1.25, 1.5, 2, 3, 5],
                            default_value=1, readonly=True, key='Play Speed'),
                   sg.Text("", size=(9, 1)),
                   sg.Button('Clear Label'), sg.Button('Label Start'), sg.Button('Event Start', key='Event Label'),
                   sg.Button('Label End')],
                  [sg.Text('Label Start: ? Label End: ?', key='StartEnd', size=(50, 1))]
                  ]

    window = sg.Window('MultiClass Video Labeler',
                       [[sg.TabGroup([[sg.Tab('Setup Page', label_dash), sg.Tab('Labeler Page', video_dash)]])]],
                       return_keyboard_events=True, use_default_focus=True)

    play = False
    frame_num = 0
    last_frame_num = -1
    event_start = None
    first = True
    while True:
        event, values = window.read(timeout=0)  # if play or first else window.read()
        first = False

        if event is None or event == 'Exit':
            break
        if event == 'Save Labels':
            video_dic['Labeler'] = values['Labeler']
            json.dump(video_dic,
                      sg.filedialog.asksaveasfile(filetypes=[('JSON File', '*.json')],
                                                  defaultextension=[('JSON File', '*.json')]))
        if event == 'Load Labels':
            video_dic = json.load(sg.filedialog.askopenfile(filetypes=[('JSON File', '*.json')],
                                                            defaultextension=[('JSON File', '*.json')]))
            print(video_dic)

        if event == 'Video Selector':
            video = videos[video_paths.index(values['Video Selector'])]
            video_name = video.get_name()
            window['Frame Slider'].update(0, range=(0, video.get_num_frames()))
            window['Frame Picker'].update('0', values=[str(i) for i in range(video.get_num_frames())])
            frame_num = 0

            video_dic[video_name] = video_dic.get(video_name, {})
            video_dic[video_name][values['EventClass']] = video_dic[video_name].get(values['EventClass'], {})
            start = video_dic[video_name][values['EventClass']].get('Label Start', '?')
            end = video_dic[video_name][values['EventClass']].get('Label End', '?')
            window['StartEnd'].update(f'Label Start: {start} Label End: {end}')

        if event == 'Frame Slider':
            try:
                temp_frame_num = int(values['Frame Slider'])
                if not 0 <= temp_frame_num < video.get_num_frames():
                    raise Exception(f'Frame Num {temp_frame_num} is out of the bounds 0 - {video.get_num_frames()}')
                frame_num = temp_frame_num
                window['Frame Picker'].update(frame_num)
            except Exception as e:
                sg.popup_error(f'Invalid Values Provided:\n{e}')
        if event == 'GO':
            try:
                temp_frame_num = int(values['Frame Picker'])
                if not 0 <= temp_frame_num < video.get_num_frames() - 1:
                    raise Exception(f'Frame Num {temp_frame_num} is out of the bounds 0 - {video.get_num_frames()}')
                frame_num = temp_frame_num
                window['Frame Slider'].update(frame_num)
            except Exception as e:
                sg.popup_error(f'Invalid Values Provided:\n{e}')
        if event == 'PlayPause' or event == 'p':
            play = not play
            window['PlayPause'].update('Pause' if play else 'Play')
        if event == 'r':
            window['Play Speed'].update(str(float(values['Play Speed']) * -1))
        if play or ((event == 'd' or event == 'a') and good_focus(window)):
            if event == 'a':
                temp_frame_num = frame_num - float(values['Play Speed'])
            else:
                temp_frame_num = frame_num + float(values['Play Speed'])
            if 0 <= temp_frame_num < video.get_num_frames():
                frame_num = temp_frame_num
            else:
                play = False
                window['PlayPause'].update('Play')

            window['Frame Picker'].update(int(frame_num))
            window['Frame Slider'].update(int(frame_num))

        if event == 'Event Label' or event == 'l':
            if window['Event Label'].GetText() == 'Clear Event':
                clear_range(frame_num, video_dic, values['Video Selector'], values['EventClass'])
            else:
                if window['Event Label'].GetText() == 'Event End':
                    append_range(event_start, frame_num, video_dic, values['Video Selector'], values['EventClass'])
                event_start = frame_num if event_start is None else None

        if event == 'EventClass':
            event_start = None

        if event == 'Label Start' or event == 'Label End':
            video_name = os.path.basename(values['Video Selector'])
            video_dic[video_name] = video_dic.get(video_name, {})
            video_dic[video_name][values['EventClass']] = video_dic[video_name].get(values['EventClass'], {})
            video_dic[video_name][values['EventClass']][event] = frame_num

            start = video_dic[video_name][values['EventClass']].get('Label Start', '?')
            end = video_dic[video_name][values['EventClass']].get('Label End', '?')
            window['StartEnd'].update(f'Label Start: {start} Label End: {end}')

        if frame_num != last_frame_num or event != '__TIMEOUT__':
            try:
                last_frame_num = frame_num

                frame = video.grab_frame_with_bparts(frame_num)

                if (event_start is not None and frame_num >= event_start) or event_in_range(frame_num, video_dic,
                                                                                            values['Video Selector'],
                                                                                            values['EventClass']):
                    frame[:, :, 0] = np.zeros([frame.shape[0], frame.shape[1]])
                    if event_start is None:
                        window['Event Label'].update('Clear Event')
                    else:
                        window['Event Label'].update('Event End')
                else:
                    window['Event Label'].update('Event Start')

                imgbytes = cv2.imencode('.png', frame)[1].tobytes()
                window['image'].update(data=imgbytes)

                video_name = os.path.basename(values['Video Selector'])

            except Exception as e:
                logging.exception(e)
                sg.popup_error(f'Error Loading Frame:\n{e}\nPlease Manually Set Valid Frame # and Try Again')

    window.close()


def event_in_range(frame_num, video_dic, video_path, event_class):
    video_name = os.path.basename(video_path)
    for start, stop in video_dic.get(video_name, {}).get(event_class, {}).get('event_ranges', []):
        if start <= frame_num < stop: return True
    return False


def clear_range(frame_num, video_dic, video_path, event_class):
    video_name = os.path.basename(video_path)
    for i, vid_range in enumerate(video_dic.get(video_name, {}).get(event_class, {}).get('event_ranges', [])):
        start, stop = vid_range
        if start <= frame_num < stop:
            video_dic[video_name][event_class]['event_ranges'].pop(i)


def append_range(event_start, event_end, video_dic, video_path, event_class):
    video_name = os.path.basename(video_path)
    video_dic[video_name] = video_dic.get(video_name, {})
    video_dic[video_name][event_class] = video_dic[video_name].get(event_class, {})
    video_dic[video_name][event_class]['event_ranges'] = video_dic[video_name][event_class].get('event_ranges', [])
    video_dic[video_name][event_class]['event_ranges'] += [(event_start, event_end)]


if __name__ == '__main__':
    sg.theme('LightGrey1')
    video_labeler(launcher())
