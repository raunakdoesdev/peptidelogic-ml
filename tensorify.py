import torch
import pandas as pd

def df_to_tensor(path_to_df, dlc_flags):
    """
    path_to_df:str - path to pickle file of pandas data frame
    dlc_flags:list - list of dlc flag strings
    """

    df = pd.read_pickle(path_to_pickle)

    tensor_list = []
    for dlc_flag in dlc_flags:
        tensor_list.append(data[dlc_flag][dlc_flag])

    return torch.cat(tensor_list, 0).unsqueeze(0)


def machine_label_to_tensor(path, num_frames):
    """
    path_to_events:str - path to events file
    num_frames:int - number of frames
    """

    events = [int(line.strip()) in open(path, 'r').readlines()]
    torch.FloatTensor([frame in events for frame in list(range(num_frames))])


def generate_ml_dataset(folder=r'D:\csv'):
    input_data_files = glob.glob(os.path.join(folder, "TEST.df.pkl").replace('TEST', '/**/*'))
    input_data_files += glob.glob(os.path.join(folder, "TEST.df.pkl").replace('TEST', '/*'))
    input_data_files.sort()

    data = {'leftpaw_prob': [], 'rightpaw_prob': [], 'machine_label': [], 'video_name': [], 'frame': []}
    for i, input_data_file in enumerate(tqdm(input_data_files)):
        df = pd.read_pickle(input_data_file)
        machine_label = [int(line.strip()) for line in open(machine_label_files[i], 'r').readlines()]

        frames = list(df['bodyparts']['coords'])

        data['frame'] += frames
        data['leftpaw_prob'] += list(df['leftpaw']['likelihood'])
        data['rightpaw_prob'] += list(df['rightpaw']['likelihood'])
        data['video_name'] += [os.path.split(input_data_file)[1].split('DeepCut')[0]] * len(frames)
        data['machine_label'] += [frame in machine_label for frame in frames]

    pd.DataFrame(data).to_pickle(r'D:\main_dataset.pkl')
