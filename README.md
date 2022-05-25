# stat_reliability_measure

This repository contains implementations of Sequential Monte Carlo methods to estimate the probability of failure of neural networks under noisy inputs.

## Requirements
The code was tested on python3.8+.
The main requirements are torch>=1.8, numpy, scipy, matplotlib and pandas.
The main requirements can be installed/checked using the command `pip install -r requirements.txt`.
The requirements for optional features can installed/checked using the command `pip install -r pip_freeze.txt`

## Running the experiments
All experiments in our paper can be reproduced using the experiments scripts.
For ImageNet experiments, data cannot be downloaded automitcally. 
Validation data for ImageNet should downloaded separately and moved to the folder `./data/ImageNet`.