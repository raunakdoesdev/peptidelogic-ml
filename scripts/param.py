import glob


class Param:

    def __init__(self):

        # flags
        self.label_dlc_on = False
        self.retrain_dlc_on = False
        self.label_clf_on = False
        self.retrain_clf_on = False
        self.evaluate_vid_on = False
        self.evaluate_cluster_on = False
        self.evaluate_matching_on = True
        self.visualize_results_on = True

        # sub flags
        self.plot_compare_human_machine_instance_on = True
        self.plot_compare_human_machine_drc_on = True
        self.plot_matching_on = True

        # paths
        self.path_to_clf_model = '../../MWT/models/xgb.save'
        self.path_to_dlc_project = '../../DLC/Retraining-BenR-2020-05-25/config.yaml'
        self.path_to_results = '../../MWT/results/'
        self.path_to_videos = '../../MWT/data/videos/'
        self.path_to_video_to_dose = '../../MWT/data/spreadsheets/06_20_2020_video_id_to_dose_virtual.xlsx'
        self.path_to_blind_summary_report = "../../MWT/data/spreadsheets/06_20_2020_SUMMARY_REPORT.xlsx"
        self.path_to_blind_key_to_video = "../../MWT/data/human_video_label/blind.json"

        # evaluate video param (todo)
        self.dlc_force = False
        self.clf_force = False
        self.eval_human_force_on = False

        # visualize results param
        self.plot_fn = 'plots.pdf'

        # blind to key code
        self.test = 'MWT'
        self.investigator = 'BW'

        # other
        self.cutoff = 50000

        # specify video ids 
        self.video_ids = [
                "BW_MWT_191104_M1_R1",
                "BW_MWT_191104_M2_R2",
                "BW_MWT_191105_M1_R2",
                "BW_MWT_191105_M3_R3",
                "BW_MWT_191107_M4_R2",
                "BW_MWT_191107_M5_R3",
                "BW_MWT_191104_M3_R1",
                "BW_MWT_191104_M1_R2",
                "BW_MWT_191104_M2_R3",
                "BW_MWT_191105_M5_R2",
                "BW_MWT_191105_M6_R3",
                "BW_MWT_191104_M2_R1",
                "BW_MWT_191104_M3_R2",
                "BW_MWT_191105_M2_R2",
                "BW_MWT_191105_M4_R2",
                "BW_MWT_191105_M5_R3",
                "BW_MWT_191104_M5_R1",
                "BW_MWT_191104_M1_R3",
                "BW_MWT_191104_M6_R3",
                "BW_MWT_191105_M2_R3",
                "BW_MWT_191107_M2_R2",
                "BW_MWT_191107_M3_R2",
                "BW_MWT_191107_M1_R3",
                "BW_MWT_191107_M2_R3"
        ]

        # validation set param
        self.paths_to_human_label_files = [
            "/home/pl/projects/pl/MWT/data/spreadsheets/MWTest_20200618_BW_1.xlsm",
            "/home/pl/projects/pl/MWT/data/spreadsheets/MWTest_20200618_BW_2.xlsm",
            "/home/pl/projects/pl/MWT/data/spreadsheets/MWTest_20200618_BW_3.xlsm",
            "/home/pl/projects/pl/MWT/data/spreadsheets/MWTest_20200618_BW_4.xlsm",
            "/home/pl/projects/pl/MWT/data/spreadsheets/MWTest_20200619_BW_1.xlsm",
            "/home/pl/projects/pl/MWT/data/spreadsheets/MWTest_20200619_BW_2.xlsm",
            "/home/pl/projects/pl/MWT/data/spreadsheets/MWTest_20200619_BW_3.xlsm",
            "/home/pl/projects/pl/MWT/data/spreadsheets/MWTest_20200619_BW_4.xlsm",
        ]