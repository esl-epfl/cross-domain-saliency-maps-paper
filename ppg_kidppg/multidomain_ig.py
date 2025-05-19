import tensorflow as tf
import numpy as np


def FourierTransform(x):
    X = tf.signal.fft(tf.cast(tf.transpose(x, perm = (0, 2, 1)), 
                              dtype = tf.complex64))
    return X

def InverseFourierTransform(X):
    x = tf.transpose(tf.cast(tf.signal.ifft(X), dtype = tf.float32), 
                     perm = (0, 2, 1))
    return x

def ComplexMultidomainIntegratedGradient(x, x_explicant, 
                                         model, 
                                         transformation, 
                                         inverse_transformation,
                                         n_iterations,
                                         output_channel):

    x_in = tf.constant(x, dtype = tf.float32)
    x_baseline = tf.constant(x_explicant, dtype = tf.float32)

    a = tf.constant(np.linspace(0, 1, n_iterations), dtype = tf.complex64)

    with tf.GradientTape() as tape:
        X_in = transformation(x_in)
        X_baseline = transformation(x_baseline)

        X_samples = X_baseline + (X_in - X_baseline) * a[:, tf.newaxis, tf.newaxis]
        tape.watch(X_samples)
        x_ = inverse_transformation(X_samples)
        y_ = model(x_)
        grads = tape.gradient(y_[:, output_channel], X_samples)
        
    S = tf.math.reduce_mean(tf.math.conj(grads), axis = 0)
    multiIG = tf.math.real((X_in[0, :] - X_baseline[0, :]) * S)
    return multiIG

def MultidomainIntegratedGradient(x, x_explicant, 
                                  model,
                                  transformation,
                                  inverse_transformation,
                                  n_iterations,
                                  output_channel):

    x_in = tf.constant(x, dtype = tf.float32)
    x_baseline = tf.constant(x_explicant, dtype = tf.float32)

    a = tf.constant(np.linspace(0, 1, n_iterations), dtype = tf.float32)

    with tf.GradientTape() as tape:
        X_in = transformation(x_in)
        X_baseline = transformation(x_baseline)

        X_samples = X_baseline + (X_in - X_baseline) * a[:, tf.newaxis, tf.newaxis]
        tape.watch(X_samples)
        x_ = inverse_transformation(X_samples)
        y_ = model(x_)
        grads = tape.gradient(y_[:, output_channel], X_samples)
        
    S = tf.math.reduce_mean(grads, axis = 0)
    multiIG = (X_in[0, :] - X_baseline[0, :]) * S
    return multiIG

def IntegratedGradient(x, x_explicant, 
                       model,
                       n_iterations,
                       output_channel):

    x_in = tf.constant(x, dtype = tf.float32)
    x_baseline = tf.constant(x_explicant, dtype = tf.float32)

    a = tf.constant(np.linspace(0, 1, n_iterations), dtype = tf.float32)

    with tf.GradientTape() as tape:
        x_samples = x_baseline + (x_in - x_baseline) * a[:, tf.newaxis, tf.newaxis]
        tape.watch(x_samples)
        y_ = model(x_samples)
        grads = tape.gradient(y_[:, output_channel], x_samples)
        
    S = tf.math.reduce_mean(grads, axis = 0)
    ig = (x_in[0, :] - x_baseline[0, :]) * S
    return ig

def FourierIntegratedGradients(x, x_explicant, 
                               model,
                               n_iterations,
                               output_channel):
    return ComplexMultidomainIntegratedGradient(x, x_explicant, 
                                                model, 
                                                FourierTransform, 
                                                InverseFourierTransform,
                                                n_iterations,
                                                output_channel)
