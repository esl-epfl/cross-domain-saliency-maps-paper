# Evaluating Cross-domain IG on TIMING benchmark 

We evaluate our method on the same datasets and benchmark methods as [*TIMING: Temporality-Aware Integrated Gradients for Time Series Explanation*](https://github.com/drumpt/TIMING). 

We have added the following scripts for running cross-domain IG:
1. [run_10perc_masking_cdig.sh](./scripts/real/run_10perc_masking_cdig.sh): To run frequency domain IG.
2. [run_10perc_masking_cepstrum_cdig.sh](./scripts/real/run_10perc_masking_cepstrum_cdig.sh): To run Cepstrum domain IG.
3. [run_10perc_masking_cdig_baseline.sh](./scripts/real/run_10perc_masking_cdig_baseline.sh): To test stability of frequency-domain IG under different baseline choices.
4. [run_10perc_masking_cepstrum_cdig_baseline.sh](./scripts/real/run_10perc_masking_cepstrum_cdig_baseline.sh): To test stability of Cepstrum IG under different baseline choices.

We have also added the following main experiment python scripts:
1. [main_cdig.py](./real/main_cdig.py): Main python script for running evaluations on frequency-domain IG.
2. [main_cdig_baseline.py](./real/main_cdig_baseline.py): Main python script for running evaluations on frequency-domain IG with different IG baselines.
2. [main_cepstrum_cdig.py](./real/main_cepstrum_cdig.py): Main python script for running evaluations on cepstrum-domain IG.
2. [main_cepstrum_cdig_baseline.py](./real/main_cepstrum_cdig_baseline.py): Main python script for running evaluations on cepstrum-domain IG with different IG baselines.

See [TIMING README](./readme_original.md) for instructions on setting up the envrinoment.

To run the scripts you need to install the ```cross-domain-saliency-maps``` library:
```pip install cross-domain-saliency-maps```.