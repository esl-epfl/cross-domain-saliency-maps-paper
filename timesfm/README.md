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

For generating the IGs:
1. ```timesfm_time_ig.py``` generates the time-domain saliencymaps.
2. ```timesfm_trend_season_ig.py``` and ```timesfm_trend_season_ig_more_demos.py``` generate the time-domain saliencymaps.
Results are saved in ```./results```.

For plotting the IG results:
1. Run ```timesfm_time_ig_plots.py``` script to plot results. 
2. Run ```timesfm_trend_season_ig_plots.py``` and ```timesfm_trend_season_ig_more_demo_plots.py``` scripts to plot results. 
All plots are saved in ```./figures```.
