from config import mne_data_path
from mne import read_evokeds
from mne.io import RawArray

def _get_raw_path(subject_id):
    return '{}/{}/{}_RAW.fif'.format(mne_data_path, subject_id, subject_id)


def get_raw_data(subject_id=None):
    if subject_id is None:
        raise RuntimeError('subject is a required parameter')

    evokeds = read_evokeds(_get_raw_path(subject_id))
    return RawArray(evokeds[0].data, evokeds[0].info)

