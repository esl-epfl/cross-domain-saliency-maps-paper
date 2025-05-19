# IG in the Independent Component Analysis Domain
Generate results and figures for Figure 4 of our manuscript.

## Installation
Create a python virtual environment and activate it:
```
python -m venv eeg_zhu_env
source ./eeg_zhu_env/bin/activate
``` 

Install requirements:
```
pip install -r requirements.txt
```

The experiments use the ```pyedflib``` which requires the ISO C99 standard
for compilation, so you might need to set:
```
export CFLAGS='-std=c99'
``` 

## Run Experiments
1. Run ```zhu_transformer_ica.py``` to calculate IG for independent component
   analysis (ICA IG). Results are saved in ```./results```.
2. Run ```zhu_transformer_ica_plot_results.py``` to plot ICA IG results.
3. Run ```eeg_ica_plot.py``` plot input sample and ICA channels of input samples.

All plots are saved in ```./figures```.

The data provided in ```./data/``` are taken from the [Siena Scalp EEG Database](https://physionet.org/content/siena-scalp-eeg/1.0.0/). 