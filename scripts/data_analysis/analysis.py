

import numpy as np 
import pandas as pd 
import json 
import matplotlib.pyplot as plt 

# defaults
plt.rcParams.update({'font.size': 10})
plt.rcParams['lines.linewidth'] = 5.0 #2.5

def main():

	# bsr = blind summary report ()
	# bkey = blind key (0,1,..,23)
	# ex video : 'BW_MWT_191105_M3_R3'

	test = 'MWT'
	investigator = 'BW' 
	path_to_bsr = "/home/pl/projects/pl/MWT/data/spreadsheets/06_20_2020_SUMMARY_REPORT.xlsx" 
	path_to_bkey_to_video = "/home/pl/projects/pl/MWT/data/human_video_label/blind.json" 
	path_to_video_to_dose = "/home/pl/projects/pl/MWT/data/spreadsheets/06_20_2020_key_to_dose.xlsx" 

	# blind key to video  
	bkey_to_video = dict() 
	with open(path_to_bkey_to_video, 'r') as j:
		bkey_to_video = json.loads(j.read())

	# video to dose 
	video_to_dose = dict()
	video_to_dose_pd = pd.read_excel(path_to_video_to_dose)
	for row in range(video_to_dose_pd.shape[0]):

		datetime = video_to_dose_pd.iat[row,1]
		year = str(datetime.year)[-2:]
		month = datetime.month
		day = datetime.day

		# make sure date and month are padded 
		if day < 9:
			day = '0{}'.format(day)
		if month < 9:
			month = '0{}'.format(month)
		
		date = '{}{}{}'.format(year,month,day)
		
		mouse_num = video_to_dose_pd.iat[row,3]
		run_num = video_to_dose_pd.iat[row,2]
		dose = video_to_dose_pd.iat[row,4]

		video_name = '{}_{}_{}_{}_R{}'.format(investigator,test,date,mouse_num,run_num) 
		video_to_dose[video_name] = dose


	# get data frames for each blind run 
	bsr_xls = pd.ExcelFile(path_to_bsr)
	bsrs = dict()
	for sheet_name in bsr_xls.sheet_names:
		bsrs[sheet_name] = pd.read_excel(bsr_xls, sheet_name).to_numpy()


	check_on = False
	if check_on:
		for sheet_name, bsr in bsrs.items():
			for row in range(video_to_dose_pd.shape[0]):

				bkey = str(bsr[row,0])
				video = bkey_to_video[bkey]
				dose = video_to_dose[video]				
				events = bsr[row,1:-1]

				print('bkey',bkey)
				print('video',video)
				print('dose',dose)
				print('events',events)

				if row == 2:
					exit()

	# put results in dict
	# 	1 : ((sheet, video), cumulative events) ... plot all videos 
	# 	2 : ((sheet, dose), (mean_events, std_events)) ... averaged over videos in same sheet
	# 	3 : ((sheet, video), (mean_events, std_events)) ... averaged over videos with same name in different sheets (sheet_name is always 1)
	# 	4 : ((sheet, dose), (mean_events, std_events)) ... averaged over all videos in all sheets (sheet_name is always 1)

	results_case = 4
	results = dict()
	if results_case == 1:
		for sheet_name, bsr in bsrs.items():
			result = dict()
			for row in range(video_to_dose_pd.shape[0]):
				bkey = str(bsr[row,0])
				video = bkey_to_video[bkey]
				events = bsr[row,1:-1]
				result[video] = np.cumsum(events)
			results[sheet_name] = result

	
	elif results_case == 2:
		for sheet_name, bsr in bsrs.items():
			result = dict()
			for row in range(video_to_dose_pd.shape[0]):

				bkey = str(bsr[row,0])
				video = bkey_to_video[bkey]
				dose = video_to_dose[video]
				events = bsr[row,1:-1]

				if dose in result.keys():
					result[dose].append(np.cumsum(events))
				else:
					result[dose] = [np.cumsum(events)]

			for dose, events_lst in result.items():
				array_result = np.asarray(result[dose]) # in ndose x nbins
				result[dose] = (np.mean(array_result,axis=0), np.std(array_result,axis=0))

			results[sheet_name] = result


	elif results_case == 3: 
		result = dict()
		for sheet_name, bsr in bsrs.items():
			for row in range(video_to_dose_pd.shape[0]):
				
				bkey = str(bsr[row,0])
				video = bkey_to_video[bkey]
				dose = video_to_dose[video]
				events = bsr[row,1:-1]
				
				if video in result.keys():
					result[video].append(np.cumsum(events))
				else:
					result[video] = [np.cumsum(events)]

		for video, events_lst in result.items():
			array_result = np.asarray(result[video]) 
			result[video] = (np.mean(array_result,axis=0), np.std(array_result,axis=0))

		results[sheet_name] = result


	elif results_case == 4: 
		result = dict()
		for sheet_name, bsr in bsrs.items():
			for row in range(video_to_dose_pd.shape[0]):
				
				bkey = str(bsr[row,0])
				video = bkey_to_video[bkey]
				dose = video_to_dose[video]
				events = bsr[row,1:-1]
				
				if dose in result.keys():
					result[dose].append(np.cumsum(events))
				else:
					result[dose] = [np.cumsum(events)]

		for dose, events_lst in result.items():
			array_result = np.asarray(result[dose]) 
			result[dose] = (np.mean(array_result,axis=0), np.std(array_result,axis=0))

		results[sheet_name] = result		


	# plotting 
	for sheet_name, result in results.items():
	
		colors = dict()
		fig,ax = plt.subplots()
		ax.set_title(sheet_name)


		if results_case == 1:
			for video, events in results[sheet_name].items():
				dose = video_to_dose[video]
				if dose in colors.keys():
					ax.plot(events,color=colors[dose])
				else:
					line = ax.plot(events)
					colors[dose] = line[0].get_color()


		elif results_case == 2:
			for dose, (mean_events,std_events) in results[sheet_name].items():
				if dose in colors.keys():
					ax.plot(events,color=colors[dose])
				else:
					line = ax.plot(mean_events)
					colors[dose] = line[0].get_color()
				ax.fill_between(range(mean_events.size), \
					mean_events-std_events,mean_events+std_events,alpha=0.5,color=colors[dose])


		elif results_case == 3:
			for video, (mean_events,std_events) in results[sheet_name].items():
				if video in colors.keys():
					ax.plot(events,color=colors[video])
				else:
					line = ax.plot(mean_events)
					colors[video] = line[0].get_color()
				ax.fill_between(range(mean_events.size), \
					mean_events-std_events,mean_events+std_events,alpha=0.5,color=colors[video])


		elif results_case == 4:
			for dose, (mean_events,std_events) in results[sheet_name].items():
				if dose in colors.keys():
					ax.plot(events,color=colors[dose])
				else:
					line = ax.plot(mean_events)
					colors[dose] = line[0].get_color()
				ax.fill_between(range(mean_events.size), \
					mean_events-std_events,mean_events+std_events,alpha=0.5,color=colors[dose])


		# colors.sort()
		for label,color in colors.items():
			ax.plot(np.nan,np.nan,color=color,label=label)

		if results_case in [1,2,4]:
			handles, labels = ax.get_legend_handles_labels()
			labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: int(t[0].split(' ')[0])))
			ax.legend(handles, labels)
		else:
			ax.legend()

		ax.grid(True)
		plt.show()


if __name__ == '__main__':
	main()