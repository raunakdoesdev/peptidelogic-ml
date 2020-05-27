# import mousenet as mn
import PySimpleGUI as sg


def project_generation():
    sg.theme('LightGrey1')
    layout = [[sg.Text('Project Name: '), sg.InputText(key='proj')], [sg.Text('Labeler Name: '), sg.InputText(key='labeler')],
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
        if event in (None, 'Exit'):
            break

    window.close()

if __name__ == '__main__':
    project_generation()
