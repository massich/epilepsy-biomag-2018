import os
import matplotlib.pyplot as plt
import mne


from config import subject_ids, subjects_dir, mne_data_path, subjects_dir
import utils


plt.close('all')

subject = subject_ids[2]
print('Processing subject: %s' % subject)
trans_fname = os.path.join(mne_data_path, subject, "%s-trans.fif" % subject)

base_dir = os.path.join(subjects_dir, "..", "original_data", subject)
bem_fname = os.path.join(base_dir, "%s-bem.fif" % subject)
fwd_surf_fname = os.path.join(base_dir, "%s-oct6-fwd.fif" % subject)

##############################################################################
# Read and plot the data

evoked = utils.get_data(subject_id=subject, maxfilter=True)

##############################################################################
# Exclude some channels

mne.viz.plot_alignment(evoked.info, trans=trans_fname, subject=subject,
                       subjects_dir=subjects_dir)

mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir)

conductivity = (0.3,)  # for single layer
# conductivity = (0.3, 0.006, 0.3)  # for three layers
model = mne.make_bem_model(subject=subject, ico=4,
                           conductivity=conductivity,
                           subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)
mne.write_bem_solution(bem_fname, bem)

src_surf = mne.setup_source_space(subject, subjects_dir=subjects_dir)
info = evoked.copy().pick_types(meg=True, eeg=False).info
fwd = mne.make_forward_solution(info, trans_fname, src_surf, bem)
mne.write_forward_solution(fwd_surf_fname, fwd, overwrite=True)
