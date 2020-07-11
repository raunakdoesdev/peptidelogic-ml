import pickle
import os
from matplotlib.backends.backend_pdf import PdfPages

import sys 
sys.path.insert(0, "../")

import mousenet as mn
import time

video_map = {'BW_MWT_191104_M4_R1': 'Vehicle(N/A)',
             'BW_MWT_191104_M2_R2': 'Vehicle(N/A)',
             'BW_MWT_191104_M5_R2': 'Vehicle(N/A)',
             'BW_MWT_191104_M3_R3': 'Vehicle(N/A)',
             'BW_MWT_191104_M5_R3': 'SKIP',
             'BW_MWT_191105_M1_R1': 'SKIP',
             'BW_MWT_191105_M2_R1': 'SKIP',
             'BW_MWT_191105_M1_R2': 'Vehicle(N/A)',
             'BW_MWT_191105_M6_R2': 'SKIP',
             'BW_MWT_191105_M1_R3': 'Vehicle(N/A)',
             'BW_MWT_191105_M3_R3': 'Vehicle(N/A)',
             'BW_MWT_191107_M5_R1': 'SKIP',
             'BW_MWT_191107_M6_R1': 'SKIP',
             'BW_MWT_191107_M4_R2': 'Vehicle(N/A)',
             'BW_MWT_191107_M5_R2': 'Vehicle(N/A)',
             'BW_MWT_191107_M5_R3': 'Vehicle(N/A)',
             'BW_MWT_191107_M6_R3': 'Vehicle(N/A)',
             'BW_MWT_191107_M4_R1': 'SKIP',
             'BW_MWT_191107_M6_R2': 'PL 100,960(3 nmole/kg)',
             'BW_MWT_191107_M4_R3': 'PL 100,960(3 nmole/kg)',
             'BW_MWT_191104_M6_R1': 'SKIP',
             'BW_MWT_191104_M3_R1': 'PL 100,960(10 nmole/kg)',
             'BW_MWT_191104_M1_R2': 'PL 100,960(10 nmole/kg)',
             'BW_MWT_191104_M4_R2': 'SKIP',
             'BW_MWT_191104_M2_R3': 'PL 100,960(10 nmole/kg)',
             'BW_MWT_191105_M3_R1': 'SKIP',
             'BW_MWT_191105_M5_R2': 'PL 100,960(10 nmole/kg)',
             'BW_MWT_191105_M6_R3': 'PL 100,960(10 nmole/kg)',
             'BW_MWT_191107_M3_R1': 'SKIP',
             'BW_MWT_191107_M1_R2': 'SKIP',
             'BW_MWT_191104_M2_R1': 'PL 100,960(30 nmole/kg)',
             'BW_MWT_191104_M3_R2': 'PL 100,960(30 nmole/kg)',
             'BW_MWT_191104_M4_R3': 'PL 100,960(30 nmole/kg)',
             'BW_MWT_191105_M5_R1': 'SKIP',
             'BW_MWT_191105_M6_R1': 'SKIP',
             'BW_MWT_191105_M2_R2': 'PL 100,960(30 nmole/kg)',
             'BW_MWT_191105_M4_R2': 'PL 100,960(30 nmole/kg)',
             'BW_MWT_191105_M5_R3': 'PL 100,960(30 nmole/kg)',
             'BW_MWT_191107_M2_R1': 'SKIP',
             'BW_MWT_191104_M1_R1': 'PL 100,960(100 nmole/kg)',
             'BW_MWT_191104_M5_R1': 'PL 100,960(100 nmole/kg)',
             'BW_MWT_191104_M6_R2': 'PL 100,960(100 nmole/kg)',
             'BW_MWT_191104_M1_R3': 'PL 100,960(100 nmole/kg)',
             'BW_MWT_191104_M6_R3': 'PL 100,960(100 nmole/kg)',
             'BW_MWT_191105_M4_R1': 'SKIP',
             'BW_MWT_191105_M3_R2': 'PL 100,960(100 nmole/kg)',
             'BW_MWT_191105_M2_R3': 'PL 100,960(100 nmole/kg)',
             'BW_MWT_191105_M4_R3': 'PL 100,960(100 nmole/kg)',
             'BW_MWT_191107_M1_R1': 'SKIP',
             'BW_MWT_191107_M2_R2': 'PL 100,960(300 nmole/kg)',
             'BW_MWT_191107_M3_R2': 'PL 100,960(300 nmole/kg)',
             'BW_MWT_191107_M1_R3': 'PL 100,960(300 nmole/kg)',
             'BW_MWT_191107_M2_R3': 'PL 100,960(300 nmole/kg)',
             'BW_MWT_191107_M3_R3': 'PL 100,960(300 nmole/kg)', }

file_name = time.strftime("/home/pl/projects/pl/MWT/results/%Y%m%d-%H%M%S.pdf")
with PdfPages(file_name) as pdf:
    clf = pickle.load(open('/home/pl/projects/pl/MWT/models/xgb.save', 'rb'))
    mn.generate_prauc_vs_data(pdf=pdf)
    mn.generate_drc(clf=clf, video_map=video_map, pdf=pdf)
    mn.generate_feature_ranking(clf=clf, pdf=pdf)
    mn.generate_temporal_feature_ranking(clf=clf, pdf=pdf)
if os.path.exists('/home/pl/Desktop/recent.pdf'):
    os.remove('/home/pl/Desktop/recent.pdf')
os.symlink(file_name, '/home/pl/Desktop/recent.pdf')
