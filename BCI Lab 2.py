# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # BCI Lab 2: P300 evoked response

# %%
# !pip install mne
# !pip install PyQt6
# !pip install PySide6
# !pip install PyQt5
# !pip install PySide2

# %%
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import os
%matplotlib qt

# %% [markdown]
# ## Loading data

# %%
fname = "P300_edf_format\group_7_p300_calibration.edf"

# print(fname)
raw = mne.io.read_raw_edf(fname, preload=True, verbose=False)


# %%
print(raw.info)
print(raw.ch_names)

# %%
# for convienence we restrict to the relevant channels for a P300
# ch_names=['Cz','C1','C2','Fz','F1','F2','Pz','P1', 'P2']
ch_names=['Cz','C3','C4','FPz','FP1','FP2','Pz']
raw.pick_channels(ch_names)
# some channel names are not standard, so we need to rename these
rename_dict={'FP1':'Fp1','FPz':'Fpz','FP2':'Fp2'}
mne.rename_channels(raw.info,rename_dict)
# for showing channel locations we need to set the montage (location) of channels
montage=mne.channels.make_standard_montage("biosemi64")
montage.plot()
raw.set_montage(montage)

# %%
raw.compute_psd(fmin=0.1,fmax=30).plot()

# %%
events, event_dict = mne.events_from_annotations(raw)
print(event_dict)
print(events[:5])

# %%
event_dict = {
     'non_target': 14,
     'target': 20
}
# different for different users
fig = mne.viz.plot_events(
    events, event_id=event_dict, sfreq=raw.info["sfreq"], first_samp=raw.first_samp
)

# %%
reject_criteria = dict(
    eeg=150e-6,  # 150 µV
    ) 
epochs = mne.Epochs(
    raw,
    events,
    event_id=event_dict,
    tmin=-0.2,
    tmax=0.7,
    baseline=(-0.1,0.0),
    preload=True,
)

# %%
target_epochs = epochs["target"]
non_target_epochs=epochs["non_target"]

# %%
target_evoked = target_epochs.average() # average over all target epochs.
non_target_evoked = non_target_epochs.average()

# %%
mne.viz.plot_compare_evokeds(
    dict(target=target_evoked, non_target=non_target_evoked),
    picks=["Cz"],
    legend="upper left",
    show_sensors="upper right",
)

# %% [markdown]
# ### In the compare plot one sees a lot of high frequency oscillations. A bandpass filter could remove these. 

# %%
raw_copy=raw.copy()

# %%
raw_copy.filter(2,15)

# %%
epochs = mne.Epochs(
    raw_copy,
    events,
    event_id=event_dict,
    tmin=-0.2,
    tmax=0.7,
    baseline=(-0.1,0.0),
    preload=True,
)

# %%
target_epochs = epochs["target"]
non_target_epochs=epochs["non_target"]
target_evoked = target_epochs.average() # average over all target epochs.
non_target_evoked = non_target_epochs.average()
mne.viz.plot_compare_evokeds(
    dict(target=target_evoked, non_target=non_target_evoked),
    picks=["Cz"],
    legend="upper left",
    show_sensors="upper right",
)

# %%
target_evoked.plot_topomap(times=[0.0, 0.1, 0.2, 0.3, 0.4])

# %%
non_target_evoked.plot_topomap(times=[0.0, 0.1, 0.2, 0.3, 0.4])

# %%
evoked_diff = mne.combine_evoked([target_evoked, non_target_evoked], weights=[1, -1])
evoked_diff.plot_topo(color="r", legend=False)

# %%
