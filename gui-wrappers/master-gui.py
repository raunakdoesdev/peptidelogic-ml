import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).parent.absolute().parent.absolute()))

import os
import shutil

import PySimpleGUI as sg

import mousenet as mn
import deeplabcut as dlc


def project_generation():
    sg.theme('LightGrey1')
    layout = [[sg.Text('Project Name: '), sg.InputText(key='proj')],
              [sg.Text('Labeler Name: '), sg.InputText(key='labeler')],
              [sg.Text('Project Directory: '),
               sg.InputText(default_text=os.path.abspath('.'), key='projdir', size=(34, 1)),
               sg.FolderBrowse()],
              [sg.Text('Body Parts:')],
              [sg.Listbox([], size=(50, 3), key='bodyparts'),
               sg.Column([[sg.Button('+', size=(3, 1))], [sg.Button('-', size=(3, 1))]])],
              [sg.Button('Create DLC Project!')]]

    window = sg.Window('DLC Model Body Part Labeler', layout)

    while True:
        event, values = window.read()
        if event == '+':
            bps = window['bodyparts'].GetListValues()
            bps.append(sg.PopupGetText(message="Enter Body Part Name", title='Get Body Part Name'))
            window['bodyparts'].update(bps)
        if event == '-' and len(values['bodyparts']) > 0:
            bps = window['bodyparts'].GetListValues()
            for bp in values['bodyparts']:
                bps.remove(bp)
            window['bodyparts'].update(bps)
        if event == 'Create DLC Project!':
            window.close()
            return mn.DLCProject(proj_name=values['proj'], labeler_name=values['labeler'],
                                 bodyparts=window['bodyparts'].GetListValues(), videos=[], proj_dir=values['projdir'])
        if event in (None, 'Exit'):
            break

    window.close()


def master_gui():
    sg.theme('LightGrey1')
    layout = [
        [sg.Text('DLC Config:'), sg.InputText(key='config'), sg.FileBrowse(), sg.Button('Create Project')],
        [sg.Button('Get Extracted Frames')],
        [sg.Button('Label Frames')]
    ]

    window = sg.Window('DLC Model Body Part Labeler', layout)
    while True:
        event, values = window.read()
        if event == 'Create Project':
            window['config'].update(project_generation().config_path)
        if event == 'Get Extracted Frames':
            frames_folder = sg.filedialog.askdirectory()
            shutil.copytree(frames_folder, os.path.join(os.path.dirname(values['config']), 'labeled-data', 'temp'))
        if event == 'Locate Config':
            dlc_config_path = sg.filedialog.askopenfilename(filetypes=[('YAML File', '*.yaml')],
                                                            defaultextension=[('YAML File', '*.yaml')])
            window['config'].update(dlc_config_path)
        if event == 'Label Frames':
            dlc.label_frames(os.path.abspath(values['config']))
        if event in (None, 'Exit'):
            break

    window.close()


if __name__ == '__main__':
    master_gui()
