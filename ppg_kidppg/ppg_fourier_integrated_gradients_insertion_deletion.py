"""
Script to perform insertion/deletion evaluation 
on the heat rate extraction model.
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

from multidomain_ig import FourierIntegratedGradientsTensor
from multidomain_ig import IntegratedGradientTensor

import pickle

import os

from tqdm import tqdm

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
        
    return model

def filter_freqs(x, freqs, n_freqs, Q = 80, fs = 32.0):
    X_filtered = x.copy()
    Q = 30
    
    filters = []
    for i in range(n_freqs):
        b, a = scipy.signal.iirnotch(w0 = freqs[i], Q=Q, fs = fs)   # returns 2nd-order (biquad) TF
        sos   = scipy.signal.tf2sos(b, a)

        filters.append(sos)
    sos = np.vstack(filters)

    X_filtered = scipy.signal.sosfiltfilt(sos, X_filtered, axis = 1)

    return X_filtered

@tf.function
def FourierIGbatch(x_batch):
    x_explicant = tf.zeros((1, 256, 1))
    n_iterations = 300
    def _one(x):
        fourier_ig = FourierIntegratedGradientsTensor(x[tf.newaxis, ...], x_explicant, model, n_iterations, 0)[0]
        return fourier_ig
    return tf.map_fn(_one, x_batch, fn_output_signature=x_batch.dtype,
                     parallel_iterations = 32)


@tf.function
def IGbatch(x_batch):
    x_explicant = tf.zeros((1, 256, 1))
    n_iterations = 300
    def _one(x):
        fourier_ig = IntegratedGradientTensor(x[tf.newaxis, ...], x_explicant, model, n_iterations, 0)
        return fourier_ig
    return tf.map_fn(_one, x_batch, fn_output_signature=x_batch.dtype,
                     parallel_iterations = 32)


os.makedirs('./results/insertion_deletion', exist_ok=True)

n_features_all = [4, 32, 64]

rng = np.random.default_rng() 

for n_features in n_features_all:
    for test_subject_id in range(1, 16):
        cf = Config(search_type = 'NAS', root = './data/')

        X, y, groups, activity = pp.preprocessing(cf.dataset, cf)


        X_test = X[groups == test_subject_id]
        y_test = y[groups == test_subject_id]


        X_test = np.transpose(X_test, axes = (0, 2, 1))


        # Create model and load pre-trained weights
        model = build_attention_model((256, 1))
        model.load_weights('./saved_models/adaptive_w_attention/model_weights/model_S' + str(int(test_subject_id)) + '.h5')

        T = 1/32.0
        N = 256
        xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

        fourierIG = FourierIGbatch(X_test)
        fourierIG = 2 * fourierIG[:, : (N//2)]

        freq_roi_indexes = np.argsort(np.abs(fourierIG), axis = 1)[:, ::-1]

        timeIG = IGbatch(X_test)
        time_roi_indexes = np.argsort(np.abs(timeIG), axis = 1)[:, ::-1][:, :(n_features * 2)]
        
        y_pred = model.predict(X_test)

        X_deletion = np.fft.rfft(X_test, axis = 1)

        X_time_deletion =  np.zeros_like(X_test)
        X_time_insertion = np.zeros_like(X_test)

        X_random_deletion =  np.fft.rfft(X_test, axis = 1)

        x_explicant = np.zeros_like(X_test[0][None, ...])

        for i in range(X_test.shape[0]):
            print("Features: ", n_features, ", subject: ", test_subject_id, "==> ", i, " / ", X_test.shape[0])
            x = X_test[i][None, ...]

            n_iterations = 300

            freqs = xf[freq_roi_indexes[i]]

            x_time_filtered = x.copy()
            x_time_filtered[:, time_roi_indexes[i], :] = 0

            X_time_insertion[i] = x - x_time_filtered
            X_time_deletion[i] = x_time_filtered

            X_deletion[i, freq_roi_indexes[i, :n_features], 0] = 0

            random_roi_indexes = rng.choice(np.arange(1, N//2), size = n_features, replace = False)
            X_random_deletion[i, random_roi_indexes[:n_features], 0] = 0


        X_deletion = np.fft.irfft(X_deletion, axis = 1)
        X_insertion =  X_test - X_deletion

        X_time_insertion = X_test - X_time_deletion

        X_random_deletion =  np.fft.irfft(X_random_deletion, axis = 1)
        X_random_insertion =  X_test - X_random_deletion

        pred_baseline = model.predict(np.zeros_like(X_test))


        y_pred_deletion = model.predict(X_deletion)
        y_pred_insertion = model.predict(X_insertion)

        y_pred_time_deletion = model.predict(X_time_deletion)
        y_pred_time_insertion = model.predict(X_time_insertion)

        y_pred_random_deletion = model.predict(X_random_deletion)
        y_pred_random_insertion = model.predict(X_random_insertion)

        results = {
            'y_pred_deletion' : y_pred_deletion,
            'y_pred_insertion' : y_pred_insertion,
            'y_pred_time_deletion' : y_pred_time_deletion,
            'y_pred_time_insertion' : y_pred_time_insertion,
            'y_pred_random_deletion' : y_pred_random_deletion,
            'y_pred_random_insertion' : y_pred_random_insertion,
            'pred_baseline' : pred_baseline,
            'y_pred' : y_pred,
            'y_test' : y_test,
        }

        with open(f'./results/insertion_deletion/S{test_subject_id}_{n_features}_features.pickle', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)