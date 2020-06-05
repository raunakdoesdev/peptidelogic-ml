import logging
import mousenet as mn

logging.getLogger().setLevel(logging.DEBUG)  # Log all info
labeled_videos = mn.json_to_videos(r'E:\Peptide Logic\Writhing', '../benv2-synced.json', mult=1)

for labeled_video in labeled_videos:
    labeled_video.calculate_mappings()