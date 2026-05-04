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

from scipy.stats import spearmanr
from scipy.stats import pearsonr


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
def FourierIGbatch(model, x_batch):
    x_explicant = tf.zeros((1, 256, 1))
    n_iterations = 300
    def _one(x):
        fourier_ig = FourierIntegratedGradientsTensor(x[tf.newaxis, ...], x_explicant, model, n_iterations, 0)[0]
        return fourier_ig
    return tf.map_fn(_one, x_batch, fn_output_signature=x_batch.dtype,
                     parallel_iterations = 32)


@tf.function
def IGbatch(model, x_batch):
    x_explicant = tf.zeros((1, 256, 1))
    n_iterations = 300
    def _one(x):
        fourier_ig = IntegratedGradientTensor(x[tf.newaxis, ...], x_explicant, model, n_iterations, 0)
        return fourier_ig
    return tf.map_fn(_one, x_batch, fn_output_signature=x_batch.dtype,
                     parallel_iterations = 32)

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

def pearson_per_sample(attr_a, attr_b):
    """
    attr_a, attr_b: arrays of shape (B, D)
    Returns per-sample Pearson correlation.
    """
    scores = []
    for a, b in zip(attr_a, attr_b):
        a = np.asarray(a).reshape(-1)
        b = np.asarray(b).reshape(-1)

        # Handle degenerate constant vectors
        if np.std(a) == 0 or np.std(b) == 0:
            scores.append(np.nan)
        else:
            scores.append(pearsonr(a, b)[0])
    return np.array(scores)


def keep_locally_stable_predictions(y_clean, y_noisy, tol_bpm=5.0):
    return np.abs(y_noisy.reshape(-1) - y_clean.reshape(-1)) <= tol_bpm

def compute_fourier_attr(model, X):
    N = X.shape[1]
    fourierIG = FourierIGbatch(model, X).numpy()   # (B, T, 1) or similar
    fourierIG = 2 * np.abs(fourierIG[:, :N//2])   # (B, N//2)
    return fourierIG

def evaluate_fourier_stability(model, X_test, xf_bpm, rng,
                               snr_levels=(20, 10, 5),
                               n_repeats=5,
                               topk_list=(4, 32, 64),
                               pred_tol_bpm=5.0):
    results = {}

    y_clean = model.predict(X_test, verbose=0).reshape(-1)
    attr_clean = compute_fourier_attr(model, X_test)

    for snr_db in snr_levels:
        print("\t SNR (db): ", snr_db)
        results[snr_db] = {
            'pearson': [],
        }
        for k in topk_list:
            results[snr_db][f'top{k}_jaccard'] = []

        for r in tqdm(range(n_repeats)):
            X_noisy = add_awgn_batch(X_test, snr_db, rng)
            y_noisy = model.predict(X_noisy, verbose=0).reshape(-1)

            keep = keep_locally_stable_predictions(y_clean, y_noisy, tol_bpm=pred_tol_bpm)
            keep_rate = np.mean(keep)

            if np.sum(keep) == 0:
                continue

            attr_noisy = compute_fourier_attr(model, X_noisy)

            clean_keep = attr_clean[keep]
            noisy_keep = attr_noisy[keep]

            results[snr_db]['pearson'].append(
                np.mean(pearson_per_sample(clean_keep, noisy_keep))
            )


    return results

os.makedirs('./results/perturbation_test', exist_ok=True)

rng = np.random.default_rng()

for test_subject_id in range(1, 16):
    print("Processing subject S" + str(int(test_subject_id)))
    # cf = Config(search_type = 'NAS', root = './data/')
    cf = Config(search_type = 'NAS', root = './data/')

    X, y, groups, activity = pp.preprocessing(cf.dataset, cf)


    X_test = X[groups == test_subject_id]
    y_test = y[groups == test_subject_id]


    X_test = np.transpose(X_test, axes = (0, 2, 1))


    # Create model and load pre-trained weights
    model = build_attention_model((256, 1))
    # model.load_weights('./saved_models/adaptive_w_attention/model_weights/model_S' + str(int(test_subject_id)) + '.h5')
    model.load_weights('./model_weights/model_S' + str(int(test_subject_id)) + '.h5')

    fs = 32.0
    N = y.size
    
    # sample spacing
    T = 1/fs
    x = np.linspace(0.0, N*T, N)
    yf = scipy.fftpack.fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2) * 60
    

    results = evaluate_fourier_stability(model = model,
                                        X_test = X_test,
                                        xf_bpm=xf,
                                        rng = rng,
                                        )

    with open(f'./results/perturbation_test/S{test_subject_id}.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)