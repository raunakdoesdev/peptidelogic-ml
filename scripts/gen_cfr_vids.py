import mousenet as mn
import os
from tqdm.auto import tqdm

videos = mn.folder_to_videos('/home/pl/projects/pl/MWT/data/videos', skip_words=('CFR',))
print(videos)


def gen_cfr(video):
    new_path = os.path.join(os.path.dirname(video.path), 'CFR' + os.path.basename(video.path))
    if not os.path.exists(new_path):
        os.system(f'ffmpeg -i {video.path} -filter:v fps=fps=30 {new_path}')


import multiprocessing as mp
import tqdm

pool = mp.Pool(processes=mp.cpu_count())
for _ in tqdm.tqdm(pool.imap_unordered(gen_cfr, videos), total=len(videos)):
    pass
