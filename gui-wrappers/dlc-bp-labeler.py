import mousenet as mn
import PySimpleGUI as sg

dlc = mn.DLCProject(config_path='/home/pl/sauhaarda/peptide_logic_refactored/dlc/'
                                'mouse_behavior_id-sauhaarda-2020-01-24/config.yaml', pcutoff=0.25)

layout = [[sg.Text('DLC Config: '), sg.InputText(), sg.Button('Browse')]]

window = sg.Window('DLC Model Body Part Labeler', layout)

while True:
    event, values = window.read()
    if event in (None, 'Exit'):
        break
        
window.close()