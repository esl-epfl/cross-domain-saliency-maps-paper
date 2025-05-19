"""
Script to visualize results created by timesfm_trend_season_ig.
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

with open('./results/timesfm_trend_season_ig_results.pickle', 'rb') as handle:
    results = pickle.load(handle)
    
t = results['t']
ig = results['ig']
delta_horizon = results['delta_horizon']
ig_delta_horizon = results['ig_delta_horizon']    
forecast_input_all = results['forecast_input_all']    
forecast_output = results['forecast_output']    
forecast_input_all_trend = results['forecast_input_all_trend']  
res_trend = results['res_trend']
res_seasonal = results['res_seasonal']  

# Plot input data and decomposition 
plt.figure(figsize = fig_size)
plt.plot(t[:512], forecast_input_all[:512], color = 'black', alpha = 1.0, 
         label = 'Timeseries')

plt.plot(t[:512], res_trend, color = 'C1', label = 'Trend')
plt.plot(t[:512], res_seasonal, color = 'C4', label = 'Seasonal')

plt.plot(t[512:], forecast_input_all[512:], color = 'black', alpha = 0.25)

plt.plot(t[512:], forecast_input_all_trend[512:], color = 'C1', alpha = 0.25)
plt.plot(t[512:], (forecast_input_all - forecast_input_all_trend)[512:], 
         alpha = 0.25, color = 'C4')


plt.legend()

if save_figure:
    plt.savefig('./figures/input_timeseries_and_decomposition.svg',
                bbox_inches = 'tight')
    
plt.figure(figsize = fig_size_appendix)

plt.plot(t[:512], res_trend, color = 'C1', label = 'Decomposed Component', linewidth = 5)
plt.plot(t[:512], forecast_input_all_trend[:512], color = 'white', 
         alpha = 1.0, label = 'Ground Truth', linestyle = 'dashed')
# plt.legend()

if save_figure:
    plt.savefig('./figures/apendix_decomposition_trend.svg',
                bbox_inches = 'tight')

plt.figure(figsize = fig_size_appendix)
plt.plot(t[:128], res_seasonal[:128], color = 'C4', label = 'Seasonal Component', linewidth = 5)
plt.plot(t[:128], (forecast_input_all - forecast_input_all_trend)[:128], color = 'white', 
         alpha = 1.0, label = 'Ground Truth', linestyle = 'dashed')
# plt.legend()

if save_figure:
    plt.savefig('./figures/apendix_decomposition_seasonal.svg',
                bbox_inches = 'tight')
    
plt.figure(figsize = fig_size_appendix)
plt.plot(t[:512], forecast_input_all[:512], color = 'black', alpha = 1.0, label = 'Timeseries')

if save_figure:
    plt.savefig('./figures/apendix_input_signal.svg',
                bbox_inches = 'tight')


trend_arrow_0 = FancyArrowPatch(posA=(t[512] - 0.125/2, 0), posB=(t[512] - 0.125/2, ig[0, 0]), 
                              arrowstyle='<|-|>', color='C1',
                              mutation_scale=12, shrinkA=0, shrinkB=0)

season_arrow_0 = FancyArrowPatch(posA=(t[512] + 0.125/2, ig[1, 0] + ig[0, 0]), posB=(t[512] + 0.125/2, ig[0, 0]), 
                               arrowstyle='<|-|>', color='C3',
                               mutation_scale=12, shrinkA=0, shrinkB=0)

trend_arrow_horizon = FancyArrowPatch(posA=(t[512 + delta_horizon] - 0.125/2, 0), posB=(t[512 + delta_horizon] - 0.125/2, ig_delta_horizon[0, 0]), 
                              arrowstyle='<|-|>', color='C1',
                              mutation_scale=12, shrinkA=0, shrinkB=0)

season_arrow_horizon = FancyArrowPatch(posA=(t[512 + delta_horizon] + 0.125/2, ig_delta_horizon[1, 0] + forecast_input_all_trend[512 + delta_horizon]), posB=(t[512 + delta_horizon] + 0.125/2, forecast_input_all_trend[512 + delta_horizon]), 
                               arrowstyle='<|-|>', color='C3',
                               mutation_scale=12, shrinkA=0, shrinkB=0)

fig, ax = plt.subplots(figsize = fig_size)
ax.plot(t[:512], forecast_input_all[:512], color = 'black')
ax.plot(t[512:], forecast_output[0, :])

ax.plot(t[512:], forecast_input_all[512:], '--', color = 'C0')
ax.plot(t[512:], forecast_input_all_trend[512:], '--',  color = 'C1', label = 'Trend')
ax.plot(t[512], forecast_output[0, 0], 'o', 
        color = 'C2', markerfacecolor = 'none', markersize = 9, 
        markeredgewidth = 3.0)

ax.plot(t[512 + delta_horizon], forecast_output[0, delta_horizon], 'o', 
        color = 'C2', markerfacecolor = 'none', markersize = 9, 
        markeredgewidth = 3.0)

ax.add_artist(trend_arrow_0)
ax.add_artist(season_arrow_0)

ax.add_artist(trend_arrow_horizon)
ax.add_artist(season_arrow_horizon)

ax.set_xlim([7, 10.1])
ax.set_ylim([0, 14])

if save_figure:
    plt.savefig('./figures/seasonal_trend_ig.svg',
                bbox_inches = 'tight')

print("Season-Trend IG in Horizon 0")
print("Trend: ", ig[0, 0])
print("Seasonality: ", ig[1, 0])
print("Prediction Error: ", np.abs(forecast_output.flatten()[0] - forecast_input_all.flatten()[-128:][0]))
print("\n====\n")

print("Season-Trend IG in Horizon " + str(int(delta_horizon)))
print("Trend: ", ig_delta_horizon[0, 0])
print("Seasonality: ", ig_delta_horizon[1, 0])
print("Prediction Error: ", np.abs(forecast_output.flatten()[delta_horizon] - forecast_input_all.flatten()[-128:][delta_horizon]))

