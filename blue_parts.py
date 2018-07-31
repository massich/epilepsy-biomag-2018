import os
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from mayavi import mlab
import mne
import nibabel as nib

from config import subject_ids, subjects_dir, mne_data_path
import utils

plt.close('all')

subject = subject_ids[0]
fig_folder = os.path.join(mne_data_path, '..', 'figures', subject)
evoked_clean_fname = os.path.join(mne_data_path, subject, '%s-ave.fif' % subject)
trans_fname = os.path.join(mne_data_path, subject, "%s-trans.fif" % subject)
print('Processing subject: %s' % subject)

##############################################################################
# Read and plot the data

evoked = utils.get_data(subject_id=subject, maxfilter=True)

##################################e############################################
# Exclude some channels

bads = {subject_ids[0]: ['EEG045', 'EEG023', 'EEG032', 'EEG024', 'EEG061',
                         'EEG020', 'EEG029', 'EEG019', 'EEG009',
                         'MEG1343', 'MEG1312', 'MEG1313', 'MEG1314',
                         'MEG1341'],
        subject_ids[1]: ['EEG033', 'EEG034',
                         'EEG058', 'EEG064','EEG036', 'EEG051',
                         'EEG023', 'EEG053','EEG049',
                         'EEG042', 'EEG055',
                        'MEG2613', 'MEG2622', 'MEG1432', 'MEG1433', 'MEG0113', #GRAD
                        'MEG0122', 'MEG0123', 'MEG0132', 'MEG0143', 'MEG1323', #GRAD
                        'MEG1332', 'MEG1423', 'MEG1512', 'MEG1513', 'MEG1522', #GRAD
                        'MEG1533', 'MEG1542', 'MEG1543', 'MEG2623', 'MEG2643', #GRAD
                         'MEG1443',
                        ],
        subject_ids[2]: ['EEG048', 'EEG063','EEG055', 'EEG062', 'EEG053', 'EEG064',
                         'MEG1421', 'MEG1431', 'MEG1331', 'MEG1321', 'MEG1311', 'MEG1341', 'MEG1411', 'MEG2611', 'MEG2421', 'MEG2641', 'MEG1441',
                         'MEG2341', 'MEG2411', # MAG but not sure if they should go in or out
                         # 'MEG1412', 'MEG1413', 'MEG1442', # GRAD
                         # 'MEG2413', 'MEG2422', 'MEG2423',
                         # 'MEG1423', 'MEG1433', 'MEG1443', 'MEG1333', 'MEG1342', 'MEG1222', 'MEG2612', 'MEG2643', 'MEG2642', 'MEG2623',
                         # 'MEG1322', 'MEG1323', 'MEG1332', 'MEG0313'],
                          'MEG1222', 'MEG1322', 'MEG1323', 'MEG1333', 'MEG1342', 'MEG1412',
                          'MEG1413', 'MEG1423', 'MEG1433', 'MEG1442', 'MEG1443', 'MEG2413',
                          'MEG2422', 'MEG2423', 'MEG2433', 'MEG2612', 'MEG2613', 'MEG2623',
                          'MEG2642',
                          'MEG1133', 'MEG1223', 'MEG1312', 'MEG1313',  # grad from no croping
                          'MEG1332', 'MEG1343', 'MEG1422', 'MEG1432',  # grad from no croping
                          'MEG2222', 'MEG2323', 'MEG2412', 'MEG2432',  # grad from no croping
                          'MEG2512', 'MEG2513', 'MEG2522', 'MEG2523',  # grad from no croping
                          'MEG2622', 'MEG2633', 'MEG2643',             # grad from no croping
                          'MEG0143', 'MEG1543', 'MEG1223', 'MEG1713',  # removed as outlier with no crop and psd analysis
        ]      }


########
# This commented section is almost surely wrong

# Function to exclude bad grad channels with high variance
def get_bad_channels(cov, perc=100):
    diag = cov.data.diagonal()
    threshold = np.percentile(diag, perc)
    idx = np.where(diag > threshold)[0]
    return idx


def _get_sensor_name_from_covariance_indx(indices, meg_type="grad"):
    return [raw.copy().pick_types(meg=meg_type).info['ch_names'][xx] for xx in indices]


########

for bad in bads[subject]:
    if bad in evoked.ch_names:
        evoked.info['bads'] += [bad]

# for idx in bad_channels_indx:
#     evoked.info['bads'] += [evoked.ch_names[idx]]
##############################################################################
# Visualize the data

time_of_interest = {subject_ids[0]: (-6, 2),
                    subject_ids[1]: (-6, 3),
                    subject_ids[2]: (None, None),
                    }
tmin, tmax = time_of_interest[subject]

evoked.crop(tmin=tmin, tmax=tmax)

evoked.save(evoked_clean_fname)

fig = evoked.plot(time_unit='s')
fig.savefig(fig_folder + '/%s_evoked.png' % subject)

maps = mne.make_field_map(evoked, trans=trans_fname, subject=subject,
                          subjects_dir=subjects_dir, ch_type='meg',
                          n_jobs=1)

# # Finally, explore several points in time
# field_map = evoked.plot_field(maps, time=1.667)

##############################################################################
# Visualize the data covariance
raw = mne.io.RawArray(evoked.data, evoked.info)
cov = mne.compute_raw_covariance(raw)
fig1, fig2 = mne.viz.plot_cov(cov, raw.info)
fig1.savefig(fig_folder + '/%s_cov1.png' % subject)
fig2.savefig(fig_folder + '/%s_cov2.png' % subject)

#   #------------------------------------------------------
#   # This is how to clean the covariance !!! do not remove
#   #------------------------------------------------------
#
# To clean the 'mag' or 'grad' variance, pick only those channels and then inspect the names
# (remember that raw.pick_chanels changes the raw object)
# raw = mne.io.RawArray(evoked.data, evoked.info)
# cov = mne.compute_raw_covariance(raw.pick_types(meg='grad'))
# mne.viz.plot_cov(cov, raw.info)
# raw.info['ch_names'][xx]  # where xx is the index number of what has ben observed in the covariance matrix

# we can also explore the raw
# raw.plot()
# raw.plot_psd()
#
#
# raw = mne.io.RawArray(evoked.data, evoked.info)
# # To clean the 'mag' or 'grad' variance, pick only those channels and then inspect the names
# cov = mne.compute_raw_covariance(raw.pick_types(meg='grad'))
# mne.viz.plot_cov(cov, raw.info)
# # raw.info['ch_names'][xx]  # where xx is the index number of what has ben observed in the covariance matrix

# raw = mne.io.RawArray(evoked.data, evoked.info)
# cov = mne.compute_raw_covariance(raw.pick_types(meg='grad'))
# bad_channels_indx = get_bad_channels(cov, perc=90)
# bad_channels = _get_sensor_name_from_covariance_indx(bad_channels_indx)
# print(bad_channels)
# 0 / 0
# _get_sensor_name_from_covariance_indx(sensors)
# #
# # we can also explore the raw
# # raw.plot()
# # raw.plot_psd()

##############################################################################
# Run ICA to remove artifacts


method = 'picard'
# method = 'fastica'

meg_maxfilter_rank = raw.copy().pick_types(meg=True).estimate_rank()
# meg_maxfilter_rank = 0.98
ica = mne.preprocessing.ICA(n_components=meg_maxfilter_rank,
                            method=method,
                            random_state=42)

ica.fit(raw.copy().pick_types(meg=True))

exclude = {subject_ids[0]: [0, 32],
           subject_ids[1]: [],
           subject_ids[2]: [],
           }
ica.exclude = exclude[subject]
ica.plot_components(ch_type='mag')
ica.plot_sources(raw)
# 0 / 0
# evoked_clean = ica.apply(evoked)
# evoked_clean.plot(time_unit='s')

# # ica.fit(raw.copy().pick_types(meg='mag'))
# # ica.fit(raw.copy().pick_types(meg='grad'))

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

ica_signal_to_reconstruct = {subject_ids[0]: 33,
                             subject_ids[1]: 35,
                             subject_ids[2]: 23,
                            }
fig = ica.plot_components(ch_type='mag', picks=[ica_signal_to_reconstruct[subject]])
fig.savefig(fig_folder + '/%s_ica_comp_topo_2d.png' % subject)
ica1 = ica.copy()
ica1.exclude = list(np.setdiff1d(np.arange(ica1.n_components_),
                                 ica_signal_to_reconstruct[subject]))
evoked_components = ica1.apply(evoked).pick_types(meg=True)
fig = evoked_components.plot()
fig.savefig(fig_folder + '/%s_evoked_ica_comp.png' % subject)

raw_tmp = mne.io.RawArray(evoked_components.data, evoked.info)
noise_cov = mne.compute_raw_covariance(raw_tmp, tmin=0., tmax=6.,
                                       method='diagonal_fixed')

trans_fname = os.path.join(subjects_dir, "..", "original_data", subject,
                           "%s-trans.fif" % subject)
bem_fname = os.path.join(subjects_dir, "..", "original_data", subject,
                         "%s-bem.fif" % subject)
t_max = evoked_components.times[np.argmax(np.abs(evoked_components.data).sum(0))]

evoked_dip = evoked_components.copy().crop(t_max, t_max)

# Finally, explore several points in time
field_map = evoked_dip.plot_field(maps, time=t_max)
field_map.scene.x_minus_view()
mlab.savefig(fig_folder + '/%s_ica_comp_topo.png' % subject)

dip, residual = mne.fit_dipole(evoked_dip,
                               noise_cov, bem_fname, trans_fname)
fig = dip.plot_locations(trans_fname, subject=subject, subjects_dir=subjects_dir)
fig.savefig(fig_folder + '/%s_dip_fit.png' % subject)

trans = mne.read_trans(trans_fname)
mni_pos = mne.head_to_mni(dip.pos, mri_head_t=trans,
                          subject=subject, subjects_dir=subjects_dir)

mri_pos = mne.transforms.apply_trans(trans, dip.pos) * 1e3
t1_fname = os.path.join(subjects_dir, subject, 'mri', 'T1.mgz')
t1 = nib.load(t1_fname)
vox2ras_tkr = t1.header.get_vox2ras_tkr()
ras2vox_tkr = linalg.inv(vox2ras_tkr)
vox2ras = t1.header.get_vox2ras()
mri_pos = mne.transforms.apply_trans(ras2vox_tkr, mri_pos)  # in vox
mri_pos = mne.transforms.apply_trans(vox2ras, mri_pos)  # in RAS

from nilearn.plotting import plot_anat
from nilearn.datasets import load_mni152_template

fig = plot_anat(t1_fname, cut_coords=mri_pos[0], title='Subj: %s' % subject)
fig.savefig(fig_folder + '/%s_mri_coord.png' % subject)

template = load_mni152_template()
fig = plot_anat(template, cut_coords=mni_pos[0], title='Subj: %s (MNI Space)' % subject)
fig.savefig(fig_folder + '/%s_mni_coord.png' % subject)
