import os
import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
import mne
from mne.minimum_norm import apply_inverse
from mne.minimum_norm import make_inverse_operator


from config import subject_ids, subjects_dir, mne_data_path
import utils

plt.close('all')

subject = subject_ids[2]
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
                        'MEG2613', 'MEG2622', 'MEG1432', 'MEG1433', 'MEG0113', #GRAD
                        'MEG0122', 'MEG0123', 'MEG0132', 'MEG0143', 'MEG1323', #GRAD
                        'MEG1332', 'MEG1423', 'MEG1512', 'MEG1513', 'MEG1522', #GRAD
                        'MEG1533', 'MEG1542', 'MEG1543', 'MEG2623', 'MEG2643', #GRAD
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
                          'MEG2642']      }



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
                    subject_ids[2]: (-4, 4),
                    }
tmin, tmax = time_of_interest[subject]

evoked.crop(tmin=tmin, tmax=tmax)

evoked.save(evoked_clean_fname)

maps = mne.make_field_map(evoked, trans=trans_fname, subject=subject,
                          subjects_dir=subjects_dir, ch_type='meg',
                          n_jobs=1)

# # Finally, explore several points in time
# field_map = evoked.plot_field(maps, time=1.667)

##############################################################################
# Visualize the data covariance
raw = mne.io.RawArray(evoked.data, evoked.info)
cov = mne.compute_raw_covariance(raw)

fwd_fname = os.path.join(subjects_dir, "..", "original_data", subject,
                         "%s-oct6-fwd.fif" % subject)

fwd = mne.read_forward_solution(fwd_fname)
inverse_operator = make_inverse_operator(evoked.info, fwd, cov, loose=0.2,
                                         depth=0.8)
stc = apply_inverse(evoked, inverse_operator)
hemi = 'rh'
vertno_max, time_max = stc.get_peak(hemi=hemi)

surfer_default_kwargs = dict(hemi=hemi,
                             subjects_dir=subjects_dir,
                             # clim=dict(kind='value', lims=[8, 12, 15]),
                             views='lateral',
                             initial_time=time_max,
                             time_unit='s',
                             size=(800, 800),
                             smoothing_steps=5,
                            time_viewer=True,
                            )

def _get_clim(min, max):
    middle = 0.5*(max-min)+min
    return dict(kind='value', lims=[min, middle, max])

brain = stc.plot(clim=_get_clim(stc.data.min(), stc.data.max()),
                 **surfer_default_kwargs)
brain.add_foci(vertno_max, coords_as_verts=True, hemi=hemi, color='blue',
              scale_factor=0.6, alpha=0.5)
