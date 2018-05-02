import os
import numpy as np
import matplotlib.pyplot as plt
import mne


from config import subject_ids, subjects_dir
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
method = 'picard'
method = 'fastica'
ica = mne.preprocessing.ICA(n_components=0.98, method=method,
                            random_state=42)
ica.fit(raw.copy().pick_types(meg=True))
# ica.exclude = [0, 32]
ica.exclude = [1, 32]
# ica.exclude = []
ica.plot_components(ch_type='mag')
ica.plot_sources(raw)

evoked_clean = ica.apply(evoked)
evoked_clean.plot(time_unit='s')
# ica.fit(raw.copy().pick_types(meg='mag'))
# ica.fit(raw.copy().pick_types(meg='grad'))

##############################################################################
# Fit dipole to dipolar ICA component (option 1 with grads only)

# evoked_components = mne.EvokedArray(ica.get_components()[:, 31:32], ica.info)
# evoked_components.pick_types(meg='grad')
# n_channels = len(evoked_components.ch_names)
# noise_cov = mne.Covariance(np.eye(n_channels), evoked_components.ch_names, [], [], 1)

# trans_fname = os.path.join(subjects_dir, "..", "original_data", subject,
#                            "%s-trans.fif" % subject)
# bem_fname = os.path.join(subjects_dir, "..", "original_data", subject,
#                          "%s-bem.fif" % subject)
# dip, residual = mne.fit_dipole(evoked_components, noise_cov, bem_fname, trans_fname)
# dip.plot_locations(trans_fname, subject=subject, subjects_dir=subjects_dir)

##############################################################################
# Fit dipole to dipolar ICA component (option 2)

ica.exclude = list(np.setdiff1d(np.arange(ica.n_components_), 31))
# ica.exclude = list(np.setdiff1d(np.arange(ica.n_components_), 29))
evoked_components = ica.apply(evoked).pick_types(meg=True)
raw_tmp = mne.io.RawArray(evoked_components.data, evoked.info)
noise_cov = mne.compute_raw_covariance(raw_tmp, tmin=0., tmax=6., method='diagonal_fixed')

trans_fname = os.path.join(subjects_dir, "..", "original_data", subject,
                           "%s-trans.fif" % subject)
bem_fname = os.path.join(subjects_dir, "..", "original_data", subject,
                         "%s-bem.fif" % subject)
t_max = evoked_components.times[np.argmax(np.abs(evoked_components.data).sum(0))]
dip, residual = mne.fit_dipole(evoked_components.copy().crop(t_max, t_max),
                               noise_cov, bem_fname, trans_fname)
dip.plot_locations(trans_fname, subject=subject, subjects_dir=subjects_dir)
