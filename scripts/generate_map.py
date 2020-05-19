import logging
import mousenet as mn

logging.getLogger().setLevel(logging.DEBUG)  # Log all info
labeled_videos = mn.json_to_videos(r'D:\Peptide Logic\Writhing', 'benv2.json', mult=1)
