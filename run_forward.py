import os
import matplotlib.pyplot as plt
import mne


from config import subject_ids, subjects_dir
import utils

plt.close('all')

subject = subject_ids[0]
print('Processing subject: %s' % subject)
trans_fname = os.path.join(subjects_dir, "..", "original_data", subject,
                           "%s-trans.fif" % subject)

##############################################################################
# Read and plot the data

evoked = utils.get_data(subject_id=subject, maxfilter=True)

##############################################################################
# Exclude some channels

mne.viz.plot_alignment(evoked.info, trans=trans_fname, subject=subject,
                       subjects_dir=subjects_dir)

