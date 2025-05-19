import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import pickle 

import scipy

from epilepsy2bids.eeg import Eeg

import os

fontsize = 11

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams.update({'font.size': fontsize})  # Set global font size
print(plt.rcParams['font.size'])  # Check if it’s actually 11

os.makedirs('./figures/', exist_ok=True)

with open('./results/ica_ig_results.pickle', 'rb') as handle:
    data = pickle.load(handle)

X = data['X']
X_ica = data['X_ica'].T
ica_ig = data['ig']

n_channels = 19
n_timepoints = X_ica.shape[1]
time = np.linspace(0, 25, n_timepoints)  

importance_scores = np.sum(ica_ig, axis = 1)

sort_indexes = np.argsort(importance_scores)[::-1]

importance_scores = importance_scores[sort_indexes]
print(importance_scores)
X_ica = X_ica[sort_indexes, :]


edf_root_folder = './data/eeg/'
edf_file = 'sub-00_ses-01_task-szMonitoring_run-02_eeg.edf'

eeg = Eeg.loadEdfAutoDetectMontage(edfFile = edf_root_folder + edf_file)

channels = eeg.channels
ica_channels = ['Ch' + str(int(i + 1)) for i in range(19)]

cm = 1/2.54

fig, ax = plt.subplots(figsize = (24.0 * cm, 16.0 * cm))

OFFSET = 2400
scale = 480.0

for i in range(n_channels):
    # Plot EEG signals
    ax.plot(time, scale * X_ica[i, ...] - i * OFFSET, 
                 color='black', linewidth = 0.25,
                 alpha = 0.85)
    ax.set_xlim([time[0], time[-1]])
    ax.spines[['top', 'right']].set_visible(False)

ax.set_yticks(list(range(-18 * OFFSET, 1, OFFSET)), ica_channels, fontsize = fontsize)    
ax.set_xlabel('Time (s)', fontsize = fontsize)

plt.tight_layout()
plt.show()
plt.savefig('./figures/ica_decomposition.svg',
            bbox_inches = 'tight')

OFFSET = 4000
scale = 30.0

fig, ax = plt.subplots(figsize = (24.0 * cm, 16.0 * cm))

for i in range(n_channels):
    ax.plot(time, scale * X[i, ...] - i * OFFSET, 
                 color='black', linewidth = 0.25,
                 alpha = 0.85)
    ax.set_xlim([time[0], time[-1]])
    ax.spines[['top', 'right']].set_visible(False)

ax.set_yticks(list(range(-18 * OFFSET, 1, OFFSET)), channels, fontsize = fontsize)    
ax.set_xlabel('Time (s)', fontsize = fontsize)

plt.tight_layout()
plt.show()
plt.savefig('./figures/eeg_channels.svg',
            bbox_inches = 'tight')