import matplotlib.pyplot as plt
import mne


from config import subject_ids
import utils

plt.close('all')

subject = subject_ids[0]
print('Processing subject: %s' % subject)

##############################################################################
# Read and plot the data

evoked = utils.get_data(subject_id=subject, maxfilter=True)

##############################################################################
# Exclude some channels

bads = ['EEG045', 'EEG023', 'EEG032', 'EEG024', 'EEG061',
        'EEG020', 'EEG029', 'EEG019', 'EEG009',
        'MEG1343', 'MEG1312', 'MEG1313', 'MEG1314', 'MEG1341']
for bad in bads:
    if bad in evoked.ch_names:
        evoked.info['bads'] += [bad]

##############################################################################
# Visualize the data
evoked.crop(tmin=-6, tmax=2)
evoked.plot(time_unit='s')

##############################################################################
# Visualize the data covariance
raw = mne.io.RawArray(evoked.data, evoked.info)
cov = mne.compute_raw_covariance(raw)
mne.viz.plot_cov(cov, raw.info)

##############################################################################
# Run ICA to remove artifacts
ica = mne.preprocessing.ICA(n_components=0.98, method='picard',
                            random_state=42)
ica.fit(raw.copy().pick_types(meg=True))
ica.exclude = [0, 32]
ica.plot_components(ch_type='mag')
ica.plot_sources(raw)

evoked_clean = ica.apply(evoked)
evoked_clean.plot(time_unit='s')
# ica.fit(raw.copy().pick_types(meg='mag'))
# ica.fit(raw.copy().pick_types(meg='grad'))
