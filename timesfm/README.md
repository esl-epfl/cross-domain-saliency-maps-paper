# Seasonal-Trend IG on time series foundation model
Generate results and figures for Figure 5 of our manuscript.

## Installation
Create a python virtual environment and activate it:
```
python -m venv timesfm_env
source ./timesfm_env/bin/activate
``` 

Install requirements:
```
pip install -r requirements.txt
```

## Run Experiments
1. Run ```timesfm_trend_season_ig.py``` script to generate Seasonal-Trend IG. Results are saved in ```./results```.
2. Run ```timesfm_trend_season_ig_plots.py``` script to plot results. Plots are saved in ```./figures```.