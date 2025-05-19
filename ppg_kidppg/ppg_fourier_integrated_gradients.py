"""
Script to generate and plot frequency-domain IG 
for heart rate extraction model KIG-PPG. 
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import scipy
import numpy as np
import seaborn as sns
from sklearn.utils import shuffle

from config import Config
from preprocessing import preprocessing_Dalia_aligned_preproc as pp

from multidomain_ig import FourierIntegratedGradients

import pickle

import os

def get_session(gpu_fraction=0.333):
    gpu_options = tf.compat.v1.GPUOptions(
            per_process_gpu_memory_fraction=gpu_fraction,
            allow_growth=True)
    return tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
tf.compat.v1.keras.backend.set_session(get_session())

tf.keras.utils.set_random_seed(0) 
tf.config.experimental.enable_op_determinism()

def plot_fft(y, fs = 32.0, linewidth = None, color = None,
             label = None, true_hr = None, true_hr_color = None,
             linestyle = None, ax = None, markersize = 12,
             markeredgewidth = 3):
    N = y.size
    
    # sample spacing
    T = 1/fs
    x = np.linspace(0.0, N*T, N)
    yf = scipy.fftpack.fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2) * 60
    
    if ax == None:
        plt.plot(xf, 2.0/N * np.abs(yf[:N//2]), linewidth = linewidth,
                color = color, label = label, linestyle = linestyle)
    else:
        ax.plot(xf, 2.0/N * np.abs(yf[:N//2]), linewidth = linewidth,
            color = color, label = label, linestyle = linestyle)
    
    if true_hr != None:
        index = np.argwhere(xf >= true_hr).flatten()[0]
        index2 = np.argwhere(xf >= 2 * true_hr).flatten()[0]
        if ax == None:
            plt.plot(xf[index], 2.0 / N * np.abs(yf[:N//2][index]), 'o',
                    markersize = markersize, color = true_hr_color, markerfacecolor = 'none',
                    markeredgewidth = markeredgewidth)

            plt.plot(xf[index2], 2.0 / N * np.abs(yf[:N//2][index2]), 'o',
                    markersize = markersize, color = true_hr_color, markerfacecolor = 'none',
                    markeredgewidth = markeredgewidth)
        else:
            ax.plot(xf[index], 2.0 / N * np.abs(yf[:N//2][index]), 'o',
                    markersize = markersize, color = true_hr_color, markerfacecolor = 'none',
                    markeredgewidth = markeredgewidth)

            ax.plot(xf[index2], 2.0 / N * np.abs(yf[:N//2][index2]), 'o',
                    markersize = markersize, color = true_hr_color, markerfacecolor = 'none',
                    markeredgewidth = markeredgewidth)

def convolution_block(input_shape, n_filters, 
                      kernel_size = 5, 
                      dilation_rate = 2,
                      pool_size = 2,
                      padding = 'causal'):
        
    mInput = tf.keras.Input(shape = input_shape)
    m = mInput
    for i in range(3):
        m = tf.keras.layers.Conv1D(filters = n_filters,
                                   kernel_size = kernel_size,
                                   dilation_rate = dilation_rate,
                                    padding = padding,
                                   activation = 'relu')(m)
        
    
    m = tf.keras.layers.AveragePooling1D(pool_size = pool_size)(m)
    m = tf.keras.layers.Dropout(rate = 0.5)(m)
        
    model = tf.keras.models.Model(inputs = mInput, outputs = m)
    
    return model



def build_attention_model(input_shape, return_attention_scores = False,
                          name = None):    
    mInput = tf.keras.Input(shape = input_shape)
    
    conv_block1 = convolution_block(input_shape, n_filters = 32,
                                    pool_size = 4)
    conv_block2 = convolution_block((64, 32), n_filters = 48)
    conv_block3 = convolution_block((32, 48), n_filters = 64)
    
    m_ppg = conv_block1(mInput)
    m_ppg = conv_block2(m_ppg)
    m_ppg = conv_block3(m_ppg)
    attention_layer = tf.keras.layers.MultiHeadAttention(num_heads = 4,
                                                         key_dim = 16,
                                                         )
    if return_attention_scores:
        m, attention_weights = attention_layer(query = m_ppg, value = m_ppg,
                                               return_attention_scores = return_attention_scores)
    else:
        m = attention_layer(query = m_ppg, value = m_ppg,
                            return_attention_scores = return_attention_scores)
    
    m = tf.keras.layers.LayerNormalization()(m)
        
    m = tf.keras.layers.Flatten()(m)
    m = tf.keras.layers.Dense(units = 32, activation = 'relu')(m)
    m = tf.keras.layers.Dense(units = 1)(m)
    
    if return_attention_scores:
        model = tf.keras.models.Model(inputs = mInput, 
                                      outputs = [m, attention_weights],
                                      name = name)
    else:
        model = tf.keras.models.Model(inputs = mInput, outputs = m,
                                      name = name)
    
    model.summary()
    
    return model

sns.set_theme()

cm = 1 / 2.54

save_figure = False
fontsize = 11

fig_size = (7 * cm, 5.5 * cm)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

plt.rc('font', size = fontsize)          # controls default text sizes
plt.rc('axes', titlesize = fontsize)     # fontsize of the axes title
plt.rc('axes', labelsize = fontsize)    # fontsize of the x and y labels
plt.rc('xtick', labelsize = fontsize)    # fontsize of the tick labels
plt.rc('ytick', labelsize = fontsize)    # fontsize of the tick labels
plt.rc('legend', fontsize = fontsize)    # legend fontsize
plt.rc('figure', titlesize = fontsize)  # fontsize of the figure title

os.makedirs('./figures/', exist_ok=True)

with open('./data/ppg_input_samples.pickle', 'rb') as handle:
    samples = pickle.load(handle)

# Plot first example with correct attributions
test_subject_id = 13

# Using sample = 464
x = samples['X_S' + str(test_subject_id)]
x_explicant = np.zeros_like(x)
y_test = samples['y_test_S' + str(test_subject_id)]

# Create model and load pre-trained weights
model = build_attention_model((256, 1))
model.load_weights('./model_weights/model_S' + str(int(test_subject_id)) + '.h5')

y_pred = model.predict(x)
error = np.abs(y_pred.flatten() - y_test.flatten())
print("Error: ", error, "(Gt: ", y_test.flatten(), ", Pred: ", y_pred.flatten(), ")")

n_iterations = 1_000
fourierIG = FourierIntegratedGradients(x, x_explicant, model, n_iterations, 0).numpy()[0]

T = 1/32.0
N = 256
xf = np.linspace(0.0, 1.0/(2.0*T), N//2) * 60

gt_hr = y_test

fig, ax1 = plt.subplots(figsize = fig_size)

color = 'C1'
ax1.set_xlabel('Freq. (BPM)')
ax1.set_ylabel('Fourier IG (BPM)')
ax1.plot(xf, fourierIG[:128] * 2, color=color, 
         linewidth = 1.75)
# ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim([-17, 47])
ax1.set_xlim([0, 600])
ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'C0'
# ax2.set_ylabel('PPG Energy', color=color)  # we already handled the x-label with ax1
plot_fft(x.flatten(), color = color, linestyle='dashed', ax = ax2, 
         true_hr = gt_hr, true_hr_color='C2', linewidth = 1.75,
         markersize = 8, markeredgewidth = 2.0)
ax2.set_yticks([])
# ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim([-17, 47])

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

plt.savefig('./figures/ppgFourierIG_low_error.svg', bbox_inches = 'tight')

###################################################
# Plot second example with problematic attributions
###################################################

test_subject_id = 9
# Use sample = 3719 from PPGDalia data

x = samples['X_S' + str(test_subject_id)]
x_explicant = np.zeros_like(x)
y_test = samples['y_test_S' + str(test_subject_id)]

# Create model and load pre-trained weights
model = build_attention_model((256, 1))
model.load_weights('./model_weights/model_S' + str(int(test_subject_id)) + '.h5')

y_pred = model.predict(x)
error = np.abs(y_pred.flatten() - y_test.flatten())
print("Error: ", error, "(Gt: ", y_test.flatten(), ", Pred: ", y_pred.flatten(), ")")

n_iterations = 1_000
fourierIG = FourierIntegratedGradients(x, x_explicant, model, n_iterations, 0).numpy()[0]

T = 1/32.0
N = 256
xf = np.linspace(0.0, 1.0/(2.0*T), N//2) * 60

gt_hr = y_test

fig, ax1 = plt.subplots(figsize = fig_size)

color = 'C1'
ax1.set_xlabel('Freq. (BPM)')
ax1.set_ylabel('Fourier IG (BPM)')
ax1.plot(xf, fourierIG[:128] * 2, color=color, 
         linewidth = 1.75)
# ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim([-10, 25])
ax1.set_xlim([0, 600])
ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'C0'
# ax2.set_ylabel('PPG Energy', color=color)  # we already handled the x-label with ax1
plot_fft(x.flatten(), color = color, linestyle='dashed', ax = ax2, 
         true_hr = gt_hr, true_hr_color='C2', linewidth = 1.75,
         markersize = 8, markeredgewidth = 2.0)
ax2.set_yticks([])
# ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim([-10, 25])

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

plt.savefig('./figures/ppgFourierIG_high_error.svg', bbox_inches = 'tight')