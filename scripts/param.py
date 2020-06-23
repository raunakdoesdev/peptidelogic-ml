

import glob 

class Param:

	def __init__(self):

		# flags
		self.label_dlc_on 			= False
		self.retrain_dlc_on 		= False
		self.label_clf_on 			= False
		self.retrain_clf_on 		= False
		self.evaluate_vid_on 		= False
		self.visualize_results_on 	= True

		# paths 
		self.path_to_clf_model 				= '../../MWT/models/xgb.save'
		self.path_to_dlc_project 			= '../../DLC/Retraining-BenR-2020-05-25/config.yaml'
		self.path_to_results 				= '../../MWT/results/' 
		self.path_to_video_to_dose 			= '../../MWT/data/spreadsheets/06_20_2020_video_id_to_dose_virtual.xlsx' 
		self.path_to_blind_summary_report 	= "../../MWT/data/spreadsheets/06_20_2020_SUMMARY_REPORT.xlsx" 
		self.path_to_bkey_to_video 			= "../../MWT/data/human_video_label/blind.json" 

		take_all_on = False
		if take_all_on: 
			# use glob to pull all mp4 and npy files 
			video_dir = '../../MWT/data/videos/'
			results_dir = '../../MWT/results/'
			self.path_to_eval_videos = [file for file in glob.glob("{}/**/*.mp4".format(video_dir))]
			self.path_to_vis_files = [file for file in glob.glob("{}/*.npy".format(results_dir))]
			
		else: 
			# specify (just the blind ones)
			self.path_to_eval_videos = [
				"../../MWT/data/videos/BW_MWT_191104_M1_R1/BW_MWT_191104_M1_R1.mp4",
				"../../MWT/data/videos/BW_MWT_191104_M2_R2/BW_MWT_191104_M2_R2.mp4",
				"../../MWT/data/videos/BW_MWT_191105_M1_R2/BW_MWT_191105_M1_R2.mp4",
				"../../MWT/data/videos/BW_MWT_191105_M3_R3/BW_MWT_191105_M3_R3.mp4",
				"../../MWT/data/videos/BW_MWT_191107_M4_R2/BW_MWT_191107_M4_R2.mp4",
				"../../MWT/data/videos/BW_MWT_191107_M5_R3/BW_MWT_191107_M5_R3.mp4",
				"../../MWT/data/videos/BW_MWT_191104_M3_R1/BW_MWT_191104_M3_R1.mp4",
				"../../MWT/data/videos/BW_MWT_191104_M1_R2/BW_MWT_191104_M1_R2.mp4",
				"../../MWT/data/videos/BW_MWT_191104_M2_R3/BW_MWT_191104_M2_R3.mp4",
				"../../MWT/data/videos/BW_MWT_191105_M5_R2/BW_MWT_191105_M5_R2.mp4",
				"../../MWT/data/videos/BW_MWT_191105_M6_R3/BW_MWT_191105_M6_R3.mp4",
				"../../MWT/data/videos/BW_MWT_191104_M2_R1/BW_MWT_191104_M2_R1.mp4",
				"../../MWT/data/videos/BW_MWT_191104_M3_R2/BW_MWT_191104_M3_R2.mp4",
				"../../MWT/data/videos/BW_MWT_191105_M2_R2/BW_MWT_191105_M2_R2.mp4",
				"../../MWT/data/videos/BW_MWT_191105_M4_R2/BW_MWT_191105_M4_R2.mp4",
				"../../MWT/data/videos/BW_MWT_191105_M5_R3/BW_MWT_191105_M5_R3.mp4",
				"../../MWT/data/videos/BW_MWT_191104_M1_R1/BW_MWT_191104_M1_R1.mp4",
				"../../MWT/data/videos/BW_MWT_191104_M5_R1/BW_MWT_191104_M5_R1.mp4",
				"../../MWT/data/videos/BW_MWT_191104_M1_R3/BW_MWT_191104_M1_R3.mp4",
				"../../MWT/data/videos/BW_MWT_191104_M6_R3/BW_MWT_191104_M6_R3.mp4",
				"../../MWT/data/videos/BW_MWT_191105_M2_R3/BW_MWT_191105_M2_R3.mp4",
				"../../MWT/data/videos/BW_MWT_191107_M2_R2/BW_MWT_191107_M2_R2.mp4",
				"../../MWT/data/videos/BW_MWT_191107_M3_R2/BW_MWT_191107_M3_R2.mp4",
				"../../MWT/data/videos/BW_MWT_191107_M1_R3/BW_MWT_191107_M1_R3.mp4",
				"../../MWT/data/videos/BW_MWT_191107_M2_R3/BW_MWT_191107_M2_R3.mp4"	
			]
			self.path_to_vis_files = [
				"../../MWT/results/BW_MWT_191104_M1_R1.npy",
				"../../MWT/results/BW_MWT_191104_M2_R2.npy",
				"../../MWT/results/BW_MWT_191105_M1_R2.npy",
				"../../MWT/results/BW_MWT_191105_M3_R3.npy",
				"../../MWT/results/BW_MWT_191107_M4_R2.npy",
				"../../MWT/results/BW_MWT_191107_M5_R3.npy",
				"../../MWT/results/BW_MWT_191104_M3_R1.npy",
				"../../MWT/results/BW_MWT_191104_M1_R2.npy",
				"../../MWT/results/BW_MWT_191104_M2_R3.npy",
				"../../MWT/results/BW_MWT_191105_M5_R2.npy",
				"../../MWT/results/BW_MWT_191105_M6_R3.npy",
				"../../MWT/results/BW_MWT_191104_M2_R1.npy",
				"../../MWT/results/BW_MWT_191104_M3_R2.npy",
				"../../MWT/results/BW_MWT_191105_M2_R2.npy",
				"../../MWT/results/BW_MWT_191105_M4_R2.npy",
				"../../MWT/results/BW_MWT_191105_M5_R3.npy",
				"../../MWT/results/BW_MWT_191104_M1_R1.npy",
				"../../MWT/results/BW_MWT_191104_M5_R1.npy",
				"../../MWT/results/BW_MWT_191104_M1_R3.npy",
				"../../MWT/results/BW_MWT_191104_M6_R3.npy",
				"../../MWT/results/BW_MWT_191105_M2_R3.npy",
				"../../MWT/results/BW_MWT_191107_M2_R2.npy",
				"../../MWT/results/BW_MWT_191107_M3_R2.npy",
				"../../MWT/results/BW_MWT_191107_M1_R3.npy",
				"../../MWT/results/BW_MWT_191107_M2_R3.npy"
			]

		# evaluate video param (todo)
		self.dlc_force = False
		self.clf_force = False 

		# visualize results param 
		self.compare_human_machine_instance = True
		self.compare_human_machine_drc = True
		self.plot_fn = 'plots.pdf'

		# blind to key code 
		self.test = 'MWT'
		self.investigator = 'BW' 