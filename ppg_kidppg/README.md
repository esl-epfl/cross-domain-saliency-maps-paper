# Frequency-Domain IG on Heart Rate extraction model
Generate results and figures for Figure 3 of our manuscript.

## Installation
Create a python virtual environment and activate it:
```
python -m venv ppg_env
source ./ppg_env/bin/activate
``` 

Install requirements:
```
pip install -r requirements.txt
```

## Run Experiments
Run ```ppg_fourier_integrated_gradients.py``` script to generate Frequency-domain IG. Plots are saved in ```./figures```.

The data provided in ```./data``` are taken from [PPGdalia](https://archive.ics.uci.edu/dataset/495/ppg+dalia) and were processed
to be compatible wiht the [```kid-ppg```](https://github.com/esl-epfl/KID-PPG-Paper) workflow.
We use samples from subjects S9 and S13. 