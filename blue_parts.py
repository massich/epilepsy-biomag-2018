from config import freesurfer_path, subject_ids
import utils

subject = subject_ids[0]
print('Processing subject: %s' % subject)

##############################################################################
# Read and plot the data

raw = utils.get_raw_data(subject_id=subject)

##############################################################################
# Exclude some channels

bads = ['EEG045', 'EEG023', 'EEG032', 'EEG024', 'EEG061',
        'EEG020', 'EEG029', 'EEG019', 'EEG009',
        'MEG1343', 'MEG1312', 'MEG1313', 'MEG1314', 'MEG1341']
for bad in bads:
    if bad in raw.ch_names:
        raw.info['bads'] += [bad]

##############################################################################
# Visualize the data

evoked = utils.get_maxfilter_data(subject_id=subject)
evoked.info['bads'] = raw.info['bads']
evoked.crop(tmin=-6, tmax=2)
evoked.plot(time_unit='s')

##############################################################################
# Visualize the data covariance
raw = mne.io.RawArray(evoked.data, evoked.info)
cov = mne.compute_raw_covariance(raw)
mne.viz.plot_cov(cov, raw.info)
