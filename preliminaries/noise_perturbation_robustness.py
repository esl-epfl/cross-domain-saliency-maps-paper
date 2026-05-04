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

# Calculate integrated gradients on time and frequency domains 

f_example_1 = np.random.normal(loc = f_mu_1, scale = f_std, size = 1)
f_example_2 = np.random.normal(loc = f_mu_2, scale = f_std, size = 1)

x1_sample = np.cos(2 * np.pi * f_example_1 * t)[None, :, None]
x2_sample = np.cos(2 * np.pi * f_example_2 * t)[None, :, None]

X1_sample = tf.abs(tf.signal.fft(x1_sample.flatten())).numpy()
X2_sample = tf.abs(tf.signal.fft(x2_sample.flatten())).numpy()


def add_awgn_batch(X, snr_db, rng):
    """
    X: (B, T, C)
    Adds per-sample AWGN at target SNR in dB.
    Preserves input dtype.
    """
    X = np.asarray(X)
    signal_power = np.mean(X**2, axis=(1, 2), keepdims=True)
    noise_power = signal_power / (10 ** (snr_db / 10.0))

    noise = rng.normal(loc=0.0, scale=1.0, size=X.shape).astype(X.dtype)
    noise = noise * np.sqrt(noise_power).astype(X.dtype)

    X_noisy = X + noise
    return X_noisy.astype(X.dtype)

rng = np.random.default_rng()

xf = scipy.fft.fftfreq(N_timepoints, 1/fs)[:N_timepoints//2]
dxf = np.diff(xf)[0]

ratios1 = []
f1_star = np.argmax(np.abs(np.fft.fft(x1_sample.flatten())[:1250]))

snrs = [20, 10, 5, 0, -5, -10, -20]
radious = 6
for snr in snrs:

    # noise = np.random.normal(0, std, size = x1_sample.shape)
    x = add_awgn_batch(x1_sample, snr, rng)
    fourierIG1 = FourierIntegratedGradients(x, np.zeros_like(x1_sample), 
                                        model,
                                        n_iterations = 300,
                                        output_channel = 1)
    fourierIG1 = fourierIG1.numpy().flatten()[:1250]
    # ratios.append(fourierIG1[f1_star] / fourierIG1.sum())
    mass = np.abs(fourierIG1)
    share = (mass[f1_star - radious : f1_star + radious].sum()) / mass.sum()
    ratios1.append(share)
ratios1 = np.array(ratios1)

ratios2 = []
f2_star = np.argmax(np.abs(np.fft.fft(x2_sample.flatten())[:1250]))

for snr in snrs:
    x = add_awgn_batch(x2_sample, snr, rng)
    fourierIG2 = FourierIntegratedGradients(x, np.zeros_like(x2_sample), 
                                        model,
                                        n_iterations = 300,
                                        output_channel = 0)
    fourierIG2 = fourierIG2.numpy().flatten()[:1250]

    mass = np.abs(fourierIG2)
    share = (mass[f2_star - radious : f2_star + radious].sum()) / mass.sum()
    ratios2.append(share)
ratios2 = np.array(ratios2)

for i in range(len(snrs)):
    print(snrs[i], ratios1[i])
