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

os.makedirs('./figures/', exist_ok=True)

with open('./results/ica_ig_results.pickle', 'rb') as handle:
    data = pickle.load(handle)

X = data['X'].T
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

channels = ['Ch' + str(int(i + 1)) for i in range(19)]

cm = 1/2.54
dpi = 96.0

width_cm = 15.0
height_cm = 8.0

width_cm = int((width_cm / 2.54) * dpi)
height_cm = int((height_cm / 2.54) * dpi)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15.0 * cm, 6.0 * cm),
                         gridspec_kw={'width_ratios': [6, 1]}, sharey=True,
                         dpi=96.0)

OFFSET = 125.0
scale = 30.0

for i in range(n_channels):
    axes[0].plot(time, scale * X_ica[i, ...] - i * OFFSET, 
                 color='black', linewidth = 0.25,
                 alpha = 0.85)
    axes[0].set_xlim([time[0], time[-1]])
    axes[0].spines[['top', 'right']].set_visible(False)
    
    axes[1].barh(y = - i * OFFSET, width=importance_scores[i], 
                 height = 70.0, color='C0')
    axes[1].set_xlim([0, max(importance_scores) * 1.1])
    axes[1].spines[['top', 'right', 'left']].set_visible(False)

axes[1].set_xticks([importance_scores.min(), importance_scores.max()],
                   [f"{importance_scores.min():.1f}", f"{importance_scores.max():.1f}"])
axes[0].set_yticks([])
axes[1].set_yticks([])
    
axes[0].set_xlabel('Time (s)')
axes[1].set_xlabel('ICA IG')

plt.tight_layout()
plt.show()
plt.savefig('./figures/channel_importance_tmp_with_bars.svg',
            bbox_inches = 'tight')