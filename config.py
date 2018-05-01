import os.path as op
from os import uname

host = uname()[1]

freesurfer_path = '/home/sik/retreat/Biomag2018/freesurfer' if host=='toothless' else '/storage/store/data/biomag_challenge/Biomag2018/freesurfer'

mne_data_path = '/home/sik/retreat/Biomag2018/original_data' if host=='toothless' else '/storage/store/data/biomag_challenge/Biomag2018/original_data'

subject_ids = ('226', '245', '251')
