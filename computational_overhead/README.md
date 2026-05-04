# Computational overhead of Cross-domain IG

Calculates the additional computational overhead due to the domain transform. 
Run the script ```test_overhead.py```. The script runs a controlled experiments on a CNN (width 64 neurons) testing the following setups:

1. Constant input size - varying depth.
2. Small depth (2 layers) - verying input size. 
3. Large depth (9 layers) - varying input size. 