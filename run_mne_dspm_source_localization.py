"""
Source localization with MNE/dSPM/sLORETA/eLORETA
=================================================

The aim of this tutorial is to teach you how to compute and apply a linear
inverse method such as MNE/dSPM/sLORETA/eLORETA on evoked/raw/epochs data.
"""
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets import sample
from mne.minimum_norm import make_inverse_operator, apply_inverse

import os
import numpy as np
import matplotlib.pyplot as plt

from config import subject_ids, subjects_dir
import utils

plt.close('all')

subject = subject_ids[2]
print('Processing subject: %s' % subject)

###############################################################################
# Process MEG data

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

time_of_interest = {subject_ids[0]: (-6, 2),
                    subject_ids[1]: (-6, 2.5),
                    subject_ids[2]: (-5, 5.2),
                    }
tmin, tmax = time_of_interest[subject]
evoked.crop(tmin=tmin, tmax=tmax)

raw = mne.io.RawArray(evoked.data, evoked.info)

picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=True,
                       exclude='bads')
baseline = (None, 0)  # means from the first instant to t = 0
reject = dict(grad=4000e-13, mag=4e-12, eog=150e-6)

# epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
#                     baseline=baseline, reject=reject)

###############################################################################
# Compute regularized noise covariance
# ------------------------------------
#
# For more details see :ref:`tut_compute_covariance`.

noise_cov = mne.compute_raw_covariance(raw)
# noise_cov = mne.compute_covariance(
#     raw, tmax=0., method=['shrunk', 'empirical'])

fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, raw.info)

###############################################################################
# Compute the evoked response
# ---------------------------
# Let's just use MEG channels for simplicity.

# evoked = epochs.average().pick_types(meg=True)
xx = raw.copy().pick_types(meg=True)
evoked = mne.EvokedArray(xx[:][0], info=xx.info)
evoked.plot()
evoked.plot_topomap(times=np.linspace(0.05, 0.15, 5), ch_type='mag',
                    time_unit='s')

# Show whitening
evoked.plot_white(noise_cov, time_unit='s')

###############################################################################
# Inverse modeling: MNE/dSPM on evoked and raw data
# -------------------------------------------------

# Read the forward solution and compute the inverse operator
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-oct-6-fwd.fif'
fwd = mne.read_forward_solution(fname_fwd)

# make an MEG inverse operator
info = evoked.info
inverse_operator = make_inverse_operator(info, fwd, noise_cov,
                                         loose=0.2, depth=0.8)
del fwd

# You can write it to disk with::
#
#     >>> from mne.minimum_norm import write_inverse_operator
#     >>> write_inverse_operator('sample_audvis-meg-oct-6-inv.fif',
#                                inverse_operator)

###############################################################################
# Compute inverse solution
# ------------------------

method = "dSPM"
snr = 3.
lambda2 = 1. / snr ** 2
stc = apply_inverse(evoked, inverse_operator, lambda2,
                    method=method, pick_ori=None)

###############################################################################
# Visualization
# -------------
# View activation time-series

plt.figure()
plt.plot(1e3 * stc.times, stc.data[::100, :].T)
plt.xlabel('time (ms)')
plt.ylabel('%s value' % method)
plt.show()

###############################################################################
# Here we use peak getter to move visualization to the time point of the peak
# and draw a marker at the maximum peak vertex.

vertno_max, time_max = stc.get_peak(hemi='rh')

subjects_dir = data_path + '/subjects'
surfer_kwargs = dict(
    hemi='rh', subjects_dir=subjects_dir,
    clim=dict(kind='value', lims=[8, 12, 15]), views='lateral',
    initial_time=time_max, time_unit='s', size=(800, 800), smoothing_steps=5)
brain = stc.plot(**surfer_kwargs)
brain.add_foci(vertno_max, coords_as_verts=True, hemi='rh', color='blue',
               scale_factor=0.6, alpha=0.5)
brain.add_text(0.1, 0.9, 'dSPM (plus location of maximal activation)', 'title',
               font_size=14)

###############################################################################
# Morph data to average brain
# ---------------------------

fs_vertices = [np.arange(10242)] * 2  # fsaverage is special this way
morph_mat = mne.compute_morph_matrix(
    'sample', 'fsaverage', stc.vertices, fs_vertices, smooth=None,
    subjects_dir=subjects_dir)
stc_fsaverage = stc.morph_precomputed('fsaverage', fs_vertices, morph_mat)
brain = stc_fsaverage.plot(**surfer_kwargs)
brain.add_text(0.1, 0.9, 'Morphed to fsaverage', 'title', font_size=20)
del stc_fsaverage

###############################################################################
# Dipole orientations
# -------------------
# The ``pick_ori`` parameter of the
# :func:`mne.minimum_norm.apply_inverse` function controls
# the orientation of the dipoles. One useful setting is ``pick_ori='vector'``,
# which will return an estimate that does not only contain the source power at
# each dipole, but also the orientation of the dipoles.

stc_vec = apply_inverse(evoked, inverse_operator, lambda2,
                        method=method, pick_ori='vector')
brain = stc_vec.plot(**surfer_kwargs)
brain.add_text(0.1, 0.9, 'Vector solution', 'title', font_size=20)
del stc_vec

###############################################################################
# Note that there is a relationship between the orientation of the dipoles and
# the surface of the cortex. For this reason, we do not use an inflated
# cortical surface for visualization, but the original surface used to define
# the source space.
#
# For more information about dipole orientations, see
# :ref:`sphx_glr_auto_tutorials_plot_dipole_orientations.py`.

###############################################################################
# Now let's look at each solver:

for mi, (method, lims) in enumerate((('dSPM', [8, 12, 15]),
                                     ('sLORETA', [3, 5, 7]),
                                     ('eLORETA', [0.75, 1.25, 1.75]),)):
    surfer_kwargs['clim']['lims'] = lims
    stc = apply_inverse(evoked, inverse_operator, lambda2,
                        method=method, pick_ori=None)
    brain = stc.plot(figure=mi, **surfer_kwargs)
    brain.add_text(0.1, 0.9, method, 'title', font_size=20)
    del stc
