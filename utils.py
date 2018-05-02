from config import mne_data_path
from mne import read_evokeds
from mne.io import RawArray

def _get_raw_path(subject_id):
    return '{}/{}/{}_RAW.fif'.format(mne_data_path, subject_id, subject_id)

def _get_maxfilter_path(subject_id):
    return '{}/{}/{}_MAXfilter-ave.fif'.format(mne_data_path, subject_id, subject_id)

def get_raw_data(subject_id=None):
    if subject_id is None:
        raise RuntimeError('subject is a required parameter')

    evokeds = read_evokeds(_get_raw_path(subject_id))
    return evokeds[0]

def get_maxfilter_data(subject_id=None):
    if subject_id is None:
        raise RuntimeError('subject is a required parameter')

    evokeds = read_evokeds(_get_maxfilter_path(subject_id))
    return evokeds[0]
