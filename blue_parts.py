from config import freesurfer_path, subject_ids
import utils
# # from library import utils
# from library.utils import get_raw_data

# subject = subject_ids[2]
subject = '226'

##############################################################################
# Read and plot the data

raw = utils.get_raw_data(subject_id=subject)

##############################################################################
# Exclude some channels

raw.info['bads'] += ['EEG045', 'EEG023', 'EEG032', 'EEG024', 'EEG061', 'MEG1343', 'MEG1312', 'MEG1313', 'MEG1314', 'MEG1341']

# ##############################################################################
# # Visually inspect the RAW data
# raw.plot(scalings='auto')

# ##############################################################################
# # Events are stored as a 2D numpy array where the first column is the time
# # instant and the last one is the event number. It is therefore easy to
# # manipulate.
# #
# # Define epochs parameters:

# tmin = -0.2  # start of each epoch (200ms before the trigger)
# tmax = 0.5  # end of each epoch (500ms after the trigger)


# ##############################################################################
# # The variable raw.info['bads'] is just a python list.
# #
# # Pick the good channels, excluding raw.info['bads']:

# picks = mne.pick_types(raw.info, meg=True, eeg=True, eog=True, ecg=True,
#                        exclude='bads')

# ##############################################################################
# # Define the baseline period:

# baseline = (None, 0)  # means from the first instant to t = 0

# ##############################################################################
# # Define peak-to-peak rejection parameters for gradiometers, magnetometers
# # and EOG:

# reject = dict(grad=4000e-13, mag=4e-12, eog=150e-6)

# ##############################################################################
# # Read epochs:

# events = mne.find_events(raw, stim_channel=['SYS101', 'SYS201'])
# epochs = mne.Epochs(raw, events=events, event_id=None, tmin=tmin, tmax=tmax, proj=True, picks=picks,
#                     baseline=None, preload=False, reject=reject)
# print(epochs)
# raw


evoked = utils.get_maxfilter_data(subject_id=subject)
# evoked.copy().crop(tmin=-6, tmax=2).plot()
evoked.info['bads'] = raw.info['bads']
evoked.copy().crop(tmin=-6, tmax=2).plot()
