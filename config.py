import os
from os import uname

host = uname()[1]
user = os.environ['USER']

if host == 'toothless':
    freesurfer_path = '/home/sik/retreat/Biomag2018/freesurfer'
    mne_data_path = '/home/sik/retreat/Biomag2018/original_data'
elif user == 'alex':
    freesurfer_path = '/Users/alex/work/data/retreat_project1/Biomag2018/freesurfer'
    mne_data_path = '/Users/alex/work/data/retreat_project1/Biomag2018/original_data'
else:
    freesurfer_path = '/storage/store/data/biomag_challenge/Biomag2018/freesurfer'
    mne_data_path = '/storage/store/data/biomag_challenge/Biomag2018/original_data'

subject_ids = ('226', '245', '251')
