import mousenet as mn
import PySimpleGUI as sg

# proj_name is not None and labeler_name is not None and videos is not None and bodyparts is not None

layout = [[sg.Text('Project Name: '), sg.InputText()], [sg.Text('Labeler Name: '), sg.InputText()],
          [sg.Listbox(['1', '2', '3'], size=(20, 3)),
           sg.Column([[sg.Button('+', size=(3, 3))], [sg.Button('-', size=(3, 3))]])]]

window = sg.Window('DLC Model Body Part Labeler', layout)

while True:
    event, values = window.read()
    if event in (None, 'Exit'):
        break

window.close()
