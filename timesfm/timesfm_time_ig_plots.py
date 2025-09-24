"""
Script to visualize results created by timesfm_time_ig.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

import seaborn as sns
import pickle

import numpy as np

import os

sns.set_theme()

cm = 1 / 2.54

save_figure = True
fontsize = 11

fig_size = (7 * cm, 6.5 * cm)
fig_size_appendix = (5.5 * cm, 4.5 * cm)


plt.rcParams.update({"font.family" : "Times New Roman"})

plt.rc('font', size = fontsize)          # controls default text sizes
plt.rc('axes', titlesize = fontsize)     # fontsize of the axes title
plt.rc('axes', labelsize = fontsize)    # fontsize of the x and y labels
plt.rc('xtick', labelsize = fontsize)    # fontsize of the tick labels
plt.rc('ytick', labelsize = fontsize)    # fontsize of the tick labels
plt.rc('legend', fontsize = fontsize)    # legend fontsize
plt.rc('figure', titlesize = fontsize)  # fontsize of the figure title

os.makedirs('./figures/', exist_ok=True)

with open('./results/timesfm_time_ig_results.pickle', 'rb') as handle:
    results = pickle.load(handle)
    
t = results['t']
ig = results['ig'][0]
delta_horizon = results['delta_horizon']
ig_delta_horizon = results['ig_delta_horizon'][0]
forecast_input_all = results['forecast_input_all']    
forecast_output = results['forecast_output']    
forecast_input_all_trend = results['forecast_input_all_trend']  

print(ig.shape)

plt.figure(figsize = fig_size)
plt.plot(t[:512], forecast_input_all[:512], color = 'C0')
plt.plot(t[:512], ig_delta_horizon, color = 'C1')

if save_figure:
    plt.savefig('./figures/time_ig.svg',
                bbox_inches = 'tight')
