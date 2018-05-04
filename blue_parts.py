import os
import numpy as np
import matplotlib.pyplot as plt
import mne


from config import subject_ids, subjects_dir
import utils

plt.close('all')

subject = subject_ids[2]
print('Processing subject: %s' % subject)

##############################################################################
# Read and plot the data

evoked = utils.get_data(subject_id=subject, maxfilter=True)

##############################################################################
# Exclude some channels

bads = {subject_ids[0]: ['EEG045', 'EEG023', 'EEG032', 'EEG024', 'EEG061',
                         'EEG020', 'EEG029', 'EEG019', 'EEG009',
                         'MEG1343', 'MEG1312', 'MEG1313', 'MEG1314',
                         'MEG1341'],
        subject_ids[1]: ['EEG033', 'EEG034',
                         'EEG058', 'EEG064','EEG036', 'EEG051',
                         'EEG023', 'EEG053','EEG049',
                        ],
        subject_ids[2]: ['EEG048', 'EEG063','EEG055', 'EEG062', 'EEG053', 'EEG064',
                         'MEG1421', 'MEG1431', 'MEG1331', 'MEG1321', 'MEG1311', 'MEG1341', 'MEG1411', 'MEG2611', 'MEG2421', 'MEG2641', 'MEG1441',
                         # 'MEG2341', 'MEG2411', # MAG but not sure if they should go in or out
                         'MEG1412', 'MEG1413', 'MEG1442', # GRAD
                         'MEG2413', 'MEG2422', 'MEG2423',
                         'MEG1423', 'MEG1433', 'MEG1443', 'MEG1333', 'MEG1342', 'MEG1222', 'MEG2612', 'MEG2643', 'MEG2642', 'MEG2623',
                        ],
        }

for bad in bads[subject]:
    if bad in evoked.ch_names:
        evoked.info['bads'] += [bad]

##############################################################################
# Visualize the data

time_of_interest = {subject_ids[0]: (-6, 2),
                    subject_ids[1]: (-6, 2.5),
                    subject_ids[2]: (-5, 5.2),
                    }
tmin, tmax = time_of_interest[subject]

evoked.crop(tmin=tmin, tmax=tmax)
evoked.plot(time_unit='s')

##############################################################################
# Visualize the data covariance
raw = mne.io.RawArray(evoked.data, evoked.info)
cov = mne.compute_raw_covariance(raw)
mne.viz.plot_cov(cov, raw.info)
#
# To clean the 'mag' or 'grad' variance, pick only those channels and then inspect the names
# cov = mne.compute_raw_covariance(raw.pick_types(meg='mag'))
# raw.info['ch_names'][xx]  # where xx is the index number of what has ben observed in the covariance matrix
#
# we can also explore the raw
# raw.plot()
# raw.plot_psd()



# raw = mne.io.RawArray(evoked.data, evoked.info)
# # To clean the 'mag' or 'grad' variance, pick only those channels and then inspect the names
# cov = mne.compute_raw_covariance(raw.pick_types(meg='grad'))
# mne.viz.plot_cov(cov, raw.info)
# # raw.info['ch_names'][xx]  # where xx is the index number of what has ben observed in the covariance matrix
# #
# # def _get_sensor_name_from_covariance_indx(indices):
# #     return [raw.info['ch_names'][xx] for xx in indices]
# sensors=[178,]
# _get_sensor_name_from_covariance_indx(sensors)
# #
# # we can also explore the raw
# # raw.plot()
# # raw.plot_psd()


##############################################################################
# Run ICA to remove artifacts
components = {subject_ids[0]: .98,
              subject_ids[1]: .95,
              subject_ids[2]: .98,
             }

method = 'picard'
method = 'fastica'
ica = mne.preprocessing.ICA(n_components=components[subject], method=method,
                            random_state=42)

ica.fit(raw.copy().pick_types(meg=True))

exclude = {subject_ids[0]: [0, 32],
           subject_ids[1]: [],
           subject_ids[2]: [],
           }
ica.exclude = exclude[subject]
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
