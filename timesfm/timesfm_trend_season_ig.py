"""
Script generates the Trend-Seasonal IG and saved
results to pickle file.

For visualizing results run timesfm_trend_season_ig_plots.py.
"""

import timesfm
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from typing import Any, Sequence
import logging
from timesfm import timesfm_base
import statsmodels
from statsmodels import tsa
from statsmodels.tsa import seasonal

import pickle

from matplotlib.patches import FancyArrowPatch

import seaborn as sns

import os

def tfm_forecast(
      tfm,
      inputs: Sequence[Any],
      freq: Sequence[int] | None = None,
      window_size: int | None = None,
      forecast_context_len: int | None = None,
      return_forecast_on_context: bool = False,
      input_index: int = 0,
      n_iterations: int = 300,
      delta_horizon: int = 0
  ) -> tuple[np.ndarray, np.ndarray]:
    """Forecasts on a list of time series.

    Args:
      inputs: list of time series forecast contexts. Each context time series
        should be in a format convertible to JTensor by `jnp.array`.
      freq: frequency of each context time series. 0 for high frequency
        (default), 1 for medium, and 2 for low. Notice this is different from
        the `freq` required by `forecast_on_df`.
      window_size: window size of trend + residual decomposition. If None then
        we do not do decomposition.
      forecast_context_len: optional max context length.
      return_forecast_on_context: True to return the forecast on the context
        when available, i.e. after the first input patch.

    Returns:
    A tuple for JTensors:
    - the mean forecast of size (# inputs, # forecast horizon),
    - the full forecast (mean + quantiles) of size
        (# inputs,  # forecast horizon, 1 + # quantiles).

    Raises:
    ValueError: If the checkpoint is not properly loaded.
    """

    if freq is None:
      logging.info("No frequency provided via `freq`. Default to high (0).")
      freq = [0] * len(inputs)

    stl = statsmodels.tsa.seasonal.STL(inputs[0], seasonal=11, period = 64)
    res = stl.fit()

    trend = res.trend
    seasonal = res.seasonal
    residual = res.resid

    trend_ts, input_padding, inp_freq, pmap_pad = tfm._preprocess([trend], freq)
    seasonal_ts, input_padding, inp_freq, pmap_pad = tfm._preprocess([seasonal], freq)
    residual_ts, input_padding, inp_freq, pmap_pad = tfm._preprocess([residual], freq)

    t_trend_ts = torch.Tensor(trend_ts[input_index * tfm.global_batch_size:(input_index + 1) *
                                        tfm.global_batch_size]).to(tfm._device)


    t_seasonal_ts = torch.Tensor(seasonal_ts[input_index * tfm.global_batch_size:(input_index + 1) *
                                        tfm.global_batch_size]).to(tfm._device)

    t_residual_ts = torch.Tensor(residual_ts[input_index * tfm.global_batch_size:(input_index + 1) *
                                        tfm.global_batch_size]).to(tfm._device)

    t_input_ts = torch.cat([t_trend_ts[..., None], t_seasonal_ts[..., None], t_residual_ts[..., None]], dim = -1)
    
    t_input_padding = torch.Tensor(
        input_padding[input_index * tfm.global_batch_size:(input_index + 1) *
                      tfm.global_batch_size]).to(tfm._device)
    t_inp_freq = torch.LongTensor(
        inp_freq[input_index * tfm.global_batch_size:(input_index + 1) *
                  tfm.global_batch_size, :]).to(tfm._device)

    coeffs = torch.ones((3, 1), dtype = torch.float32).to(tfm._device)
    coeffs_baseline =torch.zeros((3, 1), dtype = torch.float32).to(tfm._device)

    scaled_coeffs = [ coeffs_baseline + (float(i) / n_iterations) * (coeffs - coeffs_baseline) for i in range(1, n_iterations + 1)]
    
    grad_sum = 0

    for scaled_coeff in tqdm(scaled_coeffs):
        scaled_coeff.requires_grad = True
        scaled_input = torch.matmul(t_input_ts, scaled_coeff)
        mean_output, full_output = tfm._model.decode(
            input_ts=scaled_input[..., 0],
            paddings=t_input_padding,
            freq=t_inp_freq,
            horizon_len=tfm.horizon_len,
            output_patch_len=tfm.output_patch_len,
            # Returns forecasts on context for parity with the Jax version.
            return_forecast_on_context=True,
        )
        mean_output[0, tfm._horizon_start + delta_horizon].backward()
        grad_sum += scaled_coeff.grad

    grad_sum /= n_iterations
    ig = (coeffs - coeffs_baseline) * grad_sum

    if not return_forecast_on_context:
      mean_output = mean_output[:, tfm._horizon_start:, ...]
      full_output = full_output[:, tfm._horizon_start:, ...]

    return mean_output[:-pmap_pad, ...], ig.detach().cpu().numpy()

tfm = timesfm.TimesFm(
      hparams=timesfm.TimesFmHparams(
          backend="gpu",
          per_core_batch_size=32,
          horizon_len=128,
      ),
      checkpoint=timesfm.TimesFmCheckpoint(
          huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
  )

t = np.linspace(0, 8, 512)
freq = 2
phase = np.pi + np.pi/4
forecast_input = np.sin(2 * np.pi * freq * t + phase) \
    + np.sin(2 * np.pi * freq * 2 * t + phase) \

forecast_input += np.exp(t/4)
frequency_input = [0]

forecast_trend_input = np.exp(t/4)

stl = statsmodels.tsa.seasonal.STL(forecast_input, seasonal=11, period = 64)
res = stl.fit()

# Forecast and get seasonal-trend IG
delta_horizon = 97

point_forecast_, ig = tfm_forecast(
    tfm = tfm,
    inputs = [forecast_input],
    freq=frequency_input,
    n_iterations=300
)

_, ig_delta_horizon = tfm_forecast(
    tfm = tfm,
    inputs = [forecast_input],
    freq=frequency_input,
    n_iterations=300,
    delta_horizon=delta_horizon
)

print("Season-Trend IG in Horizon 0")
print("Trend: ", ig[0, 0])
print("Seasonality: ", ig[1, 0])
print("Residual: ", ig[2, 0])
print("\n====\n")
print("Season-Trend IG in Horizon " + str(int(delta_horizon)))
print("Trend: ", ig_delta_horizon[0, 0])
print("Seasonality: ", ig_delta_horizon[1, 0])
print("Residual: ", ig_delta_horizon[2, 0])

t = np.linspace(0, 10, 512 + 128)
freq = 2
forecast_input_all = np.sin(2 * np.pi * freq * t + phase) \
    + np.sin(2 * np.pi * freq * 2 * t + phase) \

forecast_input_all += np.exp(t/4)

forecast_input_all_trend = np.exp(t/4)

forecast_output = point_forecast_.detach().cpu().numpy()


stl_input_all = statsmodels.tsa.seasonal.STL(forecast_input_all, seasonal=11, period = 64)
res_input_all = stl_input_all.fit()

results = {'t' : t,
           'delta_horizon' : delta_horizon,
           'ig' : ig,
           'ig_delta_horizon':ig_delta_horizon,
           'forecast_input_all' : forecast_input_all,
           'forecast_output' : forecast_output,
           'forecast_input_all_trend' : forecast_input_all_trend,
           'res_trend' : res.trend,
           'res_seasonal' : res.seasonal
           }

os.makedirs('./results/', exist_ok=True)

with open('./results/timesfm_trend_season_ig_results.pickle', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
