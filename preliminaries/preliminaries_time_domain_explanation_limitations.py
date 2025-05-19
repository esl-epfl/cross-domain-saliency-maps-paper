import tensorflow as tf
from multidomain_ig import IntegratedGradient
from multidomain_ig import FourierIntegratedGradients
import numpy as np
import scipy

from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_rgba
from matplotlib.collections import LineCollection


import seaborn as sns

import os

tf.keras.utils.set_random_seed(0)
tf.config.experimental.enable_op_determinism()

def build_model(kernel_size, N_timepoints):
    mInput = tf.keras.Input((N_timepoints, 1))

    m = tf.keras.layers.Conv1D(filters = 2, 
                        kernel_size = kernel_size, 
                        strides = 1,
                        use_bias = False,
                        padding = 'same')(mInput)
    m = tf.keras.layers.Activation('relu')(m)
    
    m = tf.keras.layers.GlobalAveragePooling1D()(m)

    model = tf.keras.models.Model(inputs = mInput,
                                  outputs = m)

    model.compile(optimizer = 'adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics = ['accuracy'])
    
    return model

def get_colors(color, alpha_arr):
    r, g, b = to_rgb(color)
    color = [(r, g, b, alpha) for alpha in alpha_arr]
    return color

image_format = '.svg'

sns.set_theme()

cm = 1 / 2.54

save_figure = True
fontsize = 11

fig_size = (5.5 * cm, 3 * cm)

plt.rc('font', size = fontsize)          # controls default text sizes
plt.rc('axes', titlesize = fontsize)     # fontsize of the axes title
plt.rc('axes', labelsize = fontsize)    # fontsize of the x and y labels
plt.rc('xtick', labelsize = fontsize)    # fontsize of the tick labels
plt.rc('ytick', labelsize = fontsize)    # fontsize of the tick labels
plt.rc('legend', fontsize = fontsize)    # legend fontsize
plt.rc('figure', titlesize = fontsize)  # fontsize of the figure title

# plt.rcParams.update({"font.family" : "Times New Roman"})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

figures_output_folder = './figures/preliminaries/'
figures_freq_response_ig_output_folder = './figures/freq_response_ig/'

os.makedirs(figures_output_folder, exist_ok=True)
os.makedirs(figures_freq_response_ig_output_folder, exist_ok=True)

## Simulation parameters
n_samples_per_class = 2_000

f_mu_1 = 1.0
f_mu_2 = 4.0

f_middle = (f_mu_2 + f_mu_1) / 2.0

f_std = 0.5

fs = 32.0
N_sec = 8.0 * 10 #80
N_timepoints = int(fs * N_sec)


# Build model
kernel_size = 31

model = build_model(kernel_size, N_timepoints)
w1 = scipy.signal.firwin(kernel_size, cutoff = f_middle, pass_zero='highpass', fs = fs)[:, None, None] * 20
w2 = scipy.signal.firwin(kernel_size, cutoff = f_middle, pass_zero='lowpass', fs = fs)[:, None, None] * 20

w = np.concatenate([w1, w2], axis = -1)

model.layers[1].set_weights([w])


## Prepare input data
t = np.linspace(0, N_sec, N_timepoints)

f1 = np.random.normal(loc = f_mu_1, scale = f_std, size = (n_samples_per_class))
phi = np.random.uniform(0, np.pi, n_samples_per_class)

x1 = np.cos(2 * np.pi * f1[..., None] * t[None, ...] + phi[..., None])[..., None]
y1 = np.zeros((n_samples_per_class, ))

f2 = np.random.normal(loc = f_mu_2, scale = f_std, size = (n_samples_per_class))
phi = np.random.uniform(0, np.pi, n_samples_per_class)

x2 = np.cos(2 * np.pi * f2[..., None] * t[None, ...] + phi[..., None])[..., None]
y2 = np.ones((n_samples_per_class, ))
y2 = np.ones((n_samples_per_class, ))

# Plot frequency response and inputs

frequencies, w_response1 = scipy.signal.freqz(w1.flatten(), fs = fs)
frequencies, w_response2 = scipy.signal.freqz(w2.flatten(), fs = fs)

fig, ax1 = plt.subplots(figsize = fig_size)

color = 'tab:red'
# ax1.set_xlabel('Freq (Hz)')
# ax1.set_ylabel('exp', color=color)
ax1.plot(frequencies, np.abs(w_response1), color='C1')
ax1.plot(frequencies, np.abs(w_response2), color='C0')
# ax1.tick_params(axis='y', labelcolor=color)
ax1.set_yticks([])

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
# ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
sns.kdeplot(f1.flatten(), fill = True, color = 'C0', ax = ax2)
sns.kdeplot(f2.flatten(), fill = True, color = 'C1', ax = ax2)
ax2.set_ylabel('')
ax2.set_yticks([])

ax1.set_xlim([-0.75, 5.75])

plt.show()
if save_figure:
    plt.savefig(figures_output_folder + 'model_freq_response_data_dist' + image_format, bbox_inches='tight')

# Calculate integrated gradients on time and frequency domains 

f_example_1 = np.random.normal(loc = f_mu_1, scale = f_std, size = 1)
f_example_2 = np.random.normal(loc = f_mu_2, scale = f_std, size = 1)

x1_sample = np.cos(2 * np.pi * f_example_1 * t)[None, :, None]
x2_sample = np.cos(2 * np.pi * f_example_2 * t)[None, :, None]

X1_sample = tf.abs(tf.signal.fft(x1_sample.flatten())).numpy()
X2_sample = tf.abs(tf.signal.fft(x2_sample.flatten())).numpy()

ig1 = IntegratedGradient(x1_sample, np.zeros_like(x1_sample), 
                        model, 
                        n_iterations = 300, 
                        output_channel = 1)
ig1 = ig1.numpy().flatten()

fourierIG1 = FourierIntegratedGradients(x1_sample, np.zeros_like(x1_sample), 
                                       model,
                                       n_iterations = 300,
                                       output_channel = 1)
fourierIG1 = fourierIG1.numpy().flatten()

ig2 = IntegratedGradient(x2_sample, np.zeros_like(x2_sample), 
                        model, 
                        n_iterations = 300, 
                        output_channel = 0)
ig2 = ig2.numpy().flatten()

fourierIG2 = FourierIntegratedGradients(x2_sample, np.zeros_like(x2_sample), 
                                       model,
                                       n_iterations = 300,
                                       output_channel = 0)
fourierIG2 = fourierIG2.numpy().flatten()

# Plot explanations

xf = scipy.fft.fftfreq(N_timepoints, 1/fs)[:N_timepoints//2]

plt.figure(figsize = fig_size)
sns.kdeplot(f1.flatten(), fill = True, color = 'C0')
sns.kdeplot(f2.flatten(), fill = True, color = 'C1')

ymin, ymax = plt.ylim()

plt.vlines(f_example_1, ymin, ymax, 
           color = 'C0', linestyles = 'dashed',
           linewidth = 2)
plt.vlines(f_example_2, ymin, ymax, 
           color = 'C1', linestyles = 'dashed',
           linewidth = 2)

plt.xlim([-0.75, 5.75])
plt.yticks([])
plt.ylabel("")

if save_figure:
    plt.savefig(figures_output_folder + 'freq_input_distributions' + image_format, bbox_inches='tight')


plt.figure(figsize = fig_size)
plt.plot(t, x2_sample.flatten(), color = "C1", linewidth = 2)
plt.plot(t, x1_sample.flatten(), color = "C0", linewidth = 2)
plt.xlim([0, 4])
plt.yticks([])
y_t_min, y_t_max = plt.ylim()

if save_figure:
    plt.savefig(figures_output_folder + 'time_domain_input' + image_format, bbox_inches='tight')

plt.figure(figsize = fig_size)
plt.plot(xf.flatten()[:N_timepoints // 2], X1_sample[:N_timepoints // 2], color = "C0", linewidth = 2)
plt.plot(xf.flatten()[:N_timepoints // 2], X2_sample[:N_timepoints // 2], color = "C1", linewidth = 2)

y_f_min, y_f_max = plt.ylim()

plt.xlim([-0.75, 5.75])
plt.yticks([])

if save_figure:
    plt.savefig(figures_output_folder + 'frequency_domain_input' + image_format, bbox_inches='tight')

relative_ig1 = (ig1 - ig1.min()) / (ig1.max() - ig1.min())
relative_ig2 = (ig2 - ig2.min()) / (ig2.max() - ig2.min())

colors1 = get_colors("C0", relative_ig1)
colors2 = get_colors("C1", relative_ig2)

points1 = np.array([t.flatten(), x1_sample.flatten()]).T.reshape(-1, 1, 2)
segments1 = np.concatenate([points1[:-1], points1[1:]], axis=1)
lc1 = LineCollection(segments1, colors=colors1, linewidth=2)

points2 = np.array([t.flatten(), x2_sample.flatten()]).T.reshape(-1, 1, 2)
segments2 = np.concatenate([points2[:-1], points2[1:]], axis=1)
lc2 = LineCollection(segments2, colors=colors2, linewidth=2)

fig, ax = plt.subplots(figsize = fig_size)
ax.add_collection(lc2)
ax.add_collection(lc1)
ax.set_ylim([y_t_min, y_t_max])
ax.set_xlim([0, 4])
plt.yticks([])

if save_figure:
    plt.savefig(figures_output_folder + 'time_domain_ig' + '.png', dpi = 800, bbox_inches='tight')

min_alpha = 0.2

relative_ig1 = (fourierIG1 - fourierIG1.min()) / (fourierIG1.max() - fourierIG1.min())
relative_ig1[relative_ig1 < min_alpha] += min_alpha
relative_ig2 = (fourierIG2 - fourierIG2.min()) / (fourierIG2.max() - fourierIG2.min())
relative_ig2[relative_ig2 < min_alpha] += min_alpha

colors1 = get_colors("C0", relative_ig1)
colors2 = get_colors("C1", relative_ig2)

points1 = np.array([xf.flatten()[:N_timepoints // 2], X1_sample[:N_timepoints // 2]]).T.reshape(-1, 1, 2)
segments1 = np.concatenate([points1[:-1], points1[1:]], axis=1)
lc1 = LineCollection(segments1, colors=colors1, linewidth=2)

points2 = np.array([xf.flatten()[:N_timepoints // 2], X2_sample[:N_timepoints // 2]]).T.reshape(-1, 1, 2)
segments2 = np.concatenate([points2[:-1], points2[1:]], axis=1)
lc2 = LineCollection(segments2, colors=colors2, linewidth=2)

fig, ax = plt.subplots(figsize = fig_size)
ax.add_collection(lc1)
ax.add_collection(lc2)
ax.set_ylim([y_f_min, y_f_max])
ax.set_xlim([-0.75, 5.75])
ax.set_yticks([])
if save_figure:
    plt.savefig(figures_output_folder + 'freq_domain_ig' + '.png', dpi = 800, bbox_inches='tight')


min_freq = 0.1
max_freq = 15.0
delta_freq = 0.1

all_fs = np.arange(min_freq, max_freq, delta_freq)

x_response = []
alphas = []
for cur_f in all_fs:
    phi = np.random.uniform(0, 2 * np.pi, 1)
    cur_x = np.cos(2 * np.pi * cur_f * t + phi)
    cur_x = cur_x - cur_x.mean()
    x_response.append(cur_x[None, :, None])
    alphas.append(np.linalg.norm(cur_x) / np.sqrt(N_timepoints/2))

alphas = np.array(alphas).flatten()


x_baseline = np.zeros_like(x_response[0])

allFourerIG1 = []
allFourerIG2 = []
for i in range(len(all_fs)):
    fourierIGResponseCh1 = FourierIntegratedGradients(x_response[i], 
                                                x_baseline, 
                                                model,
                                                n_iterations = 10_000, 
                                                output_channel = 0)
    fourierIGResponseCh1 = fourierIGResponseCh1.numpy().flatten()
    allFourerIG1.append(fourierIGResponseCh1)

    fourierIGResponseCh2 = FourierIntegratedGradients(x_response[i], 
                                                x_baseline, 
                                                model,
                                                n_iterations = 10_000, 
                                                output_channel = 1)
    fourierIGResponseCh2 = fourierIGResponseCh2.numpy().flatten()
    allFourerIG2.append(fourierIGResponseCh2)

b1 = np.zeros(all_fs.shape)
b2 = np.zeros(all_fs.shape)

for i in range(all_fs.size):
    b1[i] = np.abs(np.sum(w1.flatten() * np.exp(-1j * 2 * np.pi * all_fs[i] * np.arange(w1.size) / fs))) #* np.sqrt(np.pi)/2
    b2[i] = np.abs(np.sum(w2.flatten() * np.exp(-1j * 2 * np.pi * all_fs[i] * np.arange(w2.size) / fs))) #* np.sqrt(np.pi)/2

norm_u1 = np.linalg.norm(b1 * alphas, ord = 2)
norm_u2 = np.linalg.norm(b2 * alphas, ord = 2)

plt.figure(figsize = fig_size)
for i in range(len(allFourerIG2)):
    plt.plot(xf, allFourerIG2[i][:t.size//2] * np.pi * 2, color = 'black')

plt.plot(all_fs, np.sqrt(np.abs(b2)**2 * alphas**2),linewidth = 3.0, color = 'C0')
plt.show()
plt.xlim([-0.75, 5.75])
plt.yticks([])

if save_figure:
    plt.savefig(figures_freq_response_ig_output_folder + 'class1_freq_response_ig' + image_format, 
                bbox_inches = 'tight')

plt.figure(figsize = fig_size)
for i in range(len(allFourerIG2)):
    plt.plot(xf, allFourerIG1[i][:t.size//2] * np.pi * 2, color = 'black')

plt.plot(all_fs, np.sqrt(np.abs(b1)**2 * alphas**2),linewidth = 3.0, color = 'C1')

plt.xlim([-0.75, 5.75])
plt.yticks([])

if save_figure:
    plt.savefig(figures_freq_response_ig_output_folder + 'class2_freq_response_ig' + image_format, 
                bbox_inches = 'tight')