import mousenet as mn

videos = mn.folder_to_videos('/home/pl/projects/pl/MWT/data/videos', skip_words=('labeled', 'CFR'), labeled=True)
dlc = mn.DLCProject(proj_name='kmeans', videos=videos, labeler_name='temp', bodyparts=['temp'])

# dlc = mn.DLCProject(config_path='/home/pl/projects/pl/code/scripts/temp-temp-2020-06-18/config.yaml')
dlc.extract_frames()
