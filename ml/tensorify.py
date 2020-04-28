import pandas as pd
import torch
import glob
import os
from tqdm import tqdm
import pickle


def df_to_tensor(path_to_df, dlc_flags):
    """
    path_to_df:str - path to pickle file of pandas data frame
    dlc_flags:list - list of deeplabcut flag strings
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


def generate_ml_dataset(folder=r'D:\csv', target=None):
    data = {'leftpaw_prob': [], 'rightpaw_prob': [], 'machine_label': [], 'video_name': [], 'frame': []}
    for i, input_data_file in enumerate(tqdm(input_data_files)):
        if target is not None and target not in input_data_file:
            continue
        df = pd.read_pickle(input_data_file)
        machine_label = [int(line.strip()) for line in open(machine_label_files[i], 'r').readlines()]

        frames = list(df['bodyparts']['coords'])

        data['frame'] += frames
        data['leftpaw_prob'] += list(df['leftpaw']['likelihood'])
        data['rightpaw_prob'] += list(df['rightpaw']['likelihood'])
        data['video_name'] += [os.path.split(input_data_file)[1].split('DeepCut')[0]] * len(frames)
        data['machine_label'] += [frame in machine_label for frame in frames]

    pd.DataFrame(data).to_pickle(r'D:\main_dataset.pkl')


def get_file_paths(folder, ext):
    files = glob.glob(os.path.join(folder, f"PLACEHOLDER.{ext}").replace('PLACEHOLDER', '/**/*'))
    files += glob.glob(os.path.join(folder, f"PLACEHOLDER.{ext}").replace('PLACEHOLDER', '/*'))
    files.sort()
    return files


def get_dist(df, body_part_1, body_part_2):
    x_dist = df[body_part_1]['x'] - df[body_part_2]['x']
    y_dist = df[body_part_1]['y'] - df[body_part_2]['y']
    return (x_dist ** 2 + y_dist ** 2) ** (1 / 2)


def generate_dataset(folder='/home/pl/csv', target=None, out='/home/pl/sauhaarda/deeplearning/neck_dist.pkl'):
    input_paths = get_file_paths(folder, 'df.pkl')
    machine_label_paths = get_file_paths(folder, 'events')
    dataset = []
    for i, input_path in enumerate(tqdm(input_paths)):
        if target is not None and target not in input_paths:
            continue
        dataset.append(create_batch(input_path, machine_label_paths[i]))

    pickle.dump(dataset, open(out, 'wb'))


def create_batch(dl_path, label_path, human=True, mult=1., frame_range=None):
    input_file = pd.read_pickle(dl_path)
    tensor_list = [input_file['leftpaw']['likelihood'], input_file['rightpaw']['likelihood']]
    # get_dist(input_file, 'leftpaw', 'neck'), get_dist(input_file, 'rightpaw', 'neck')]
    tensor_list = [torch.FloatTensor(input_tensor_row).unsqueeze(0) for input_tensor_row in tensor_list]
    input_tensor = torch.cat(tensor_list, 0).unsqueeze(0)

    # Create Label Files
    if human:
        positive_labels = sorted(list(pickle.load(open(label_path, 'rb'))))
        positive_labels = [round(positive_label * mult) for positive_label in positive_labels]
    else:
        positive_labels = [int(line.strip()) for line in open(label_path, 'r').readlines()]

    if frame_range is not None:
        input_tensor = input_tensor[:, :, frame_range[0]:frame_range[1]]

    label_tensor = torch.FloatTensor([frame in positive_labels for frame in range(frame_range[0], frame_range[1])])
    return input_tensor, label_tensor


if __name__ == '__main__':
    pickle.dump([create_batch('BW_MIT_200318_M6_R_3DeepCut_resnet50_mouse_behavior_idJan24shuffle1_200000.df.pkl',
                              'BW_MIT_200318_R3_M6.human', frame_range=[0, 2500], mult=30 / 29.981110061670094)],
                open('human_label_dataset.pkl', 'wb'))
    pickle.dump([create_batch('BW_MIT_200318_M6_R_3DeepCut_resnet50_mouse_behavior_idJan24shuffle1_200000.df.pkl',
                              'BW_MIT_200318_R3_M6.human', frame_range=[2500, 5000], mult=30 / 29.981110061670094)],
                open('human_validation_dataset.pkl', 'wb'))
