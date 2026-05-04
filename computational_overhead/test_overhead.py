import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import scipy
import numpy as np
from sklearn.utils import shuffle


from multidomain_ig import FourierIntegratedGradientsTensor
from multidomain_ig import IntegratedGradientTensor


from tqdm import tqdm

from scipy.stats import spearmanr

def get_session(gpu_fraction=0.333):
    gpu_options = tf.compat.v1.GPUOptions(
            per_process_gpu_memory_fraction=gpu_fraction,
            allow_growth=True)
    return tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
tf.compat.v1.keras.backend.set_session(get_session())

tf.keras.utils.set_random_seed(0) 
tf.config.experimental.enable_op_determinism()

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

def build_model(input_shape, channels, depth, kernel_size = 5):
    mInput = tf.keras.Input(input_shape)

    m = mInput

    for i in range(depth):
        m = tf.keras.layers.Conv1D(filters = channels,
                                   kernel_size = kernel_size,
                                    padding = 'same',
                                   activation = 'relu')(m)

    m = tf.keras.layers.GlobalAveragePooling1D()(m)
    m = tf.keras.layers.Dense(1)(m)

    model = tf.keras.models.Model(inputs = mInput, outputs = m)

    return model

def build_input(shape):
    x = np.random.normal(0, 1, shape)
    
    return x

import time
import statistics

def time_attribution(model, x, x_explicant):
    return IntegratedGradientTensor(x, x_explicant, model, 300, 0)

def frequency_attribution(model, x, x_explicant):
    return FourierIntegratedGradientsTensor(x, x_explicant, model, 300, 0)

def benchmark_fn(fn, model, x, warmup=20, iters=100):
    x_explicant = tf.zeros(x.shape, dtype = tf.float32)
    # Warm-up: excludes tracing/initial setup from the benchmark
    for _ in range(warmup):
        _ = fn(model, x, x_explicant)
        tf.test.experimental.sync_devices()

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        y = fn(model, x, x_explicant)
        tf.test.experimental.sync_devices()   # critical on GPU
        end = time.perf_counter()
        times.append(end - start)

    return {
        "mean_ms": 1000 * statistics.mean(times),
        "std_ms": 1000 * statistics.pstdev(times),
        "min_ms": 1000 * min(times),
        "max_ms": 1000 * max(times),
        "n": iters,
    }

# Overhead vs depth
input_shape = (1, 256, 1)
x = build_input(input_shape)
depths = [3, 6, 9, 12]

igs = []
freqs = []

for depth in depths:
    model = build_model(input_shape[1:], 64, depth)

    ig_time = benchmark_fn(time_attribution, model, tf.constant(x, dtype = tf.float32))
    igs.append(ig_time['mean_ms'])
    freq_ig_time = benchmark_fn(frequency_attribution, model, tf.constant(x, dtype = tf.float32))
    freqs.append(freq_ig_time['mean_ms'])

igs_vs_depth = np.array(igs)
freqs_vs_depth = np.array(freqs)

# Overhead vs input size (big model)
depth = 9
input_sizes = [256, 512, 1024, 2048, 2048 * 2]

igs = []
freqs = []

for input_size in input_sizes:
    input_shape = (1, input_size, 1)
    x = build_input(input_shape)

    model = build_model(input_shape[1:], 64, depth)

    ig_time = benchmark_fn(time_attribution, model, tf.constant(x, dtype = tf.float32))
    igs.append(ig_time['mean_ms'])
    freq_ig_time = benchmark_fn(frequency_attribution, model, tf.constant(x, dtype = tf.float32))
    freqs.append(freq_ig_time['mean_ms'])

igs_vs_input_size_big = np.array(igs)
freqs_vs_input_size_big = np.array(freqs)

# Overhead vs input size (small model)
depth = 2
input_sizes = [256, 512, 1024, 2048, 2048 * 2]

igs = []
freqs = []

for input_size in input_sizes:
    input_shape = (1, input_size, 1)
    x = build_input(input_shape)

    model = build_model(input_shape[1:], 64, depth)

    ig_time = benchmark_fn(time_attribution, model, tf.constant(x, dtype = tf.float32))
    igs.append(ig_time['mean_ms'])
    freq_ig_time = benchmark_fn(frequency_attribution, model, tf.constant(x, dtype = tf.float32))
    freqs.append(freq_ig_time['mean_ms'])

igs_vs_input_size_small = np.array(igs)
freqs_vs_input_size_small = np.array(freqs)

print((freqs_vs_depth - igs_vs_depth) / freqs_vs_depth)
print((freqs_vs_input_size_small - igs_vs_input_size_small)/freqs_vs_input_size_small)
print((freqs_vs_input_size_big - igs_vs_input_size_big)/freqs_vs_input_size_big)