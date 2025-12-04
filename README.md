# Data Release for "The multiscale non-Gaussian statistics of free-running 1000-member general circulation model ensembles"

This repository contains data and scripts used to generate the figures and analysis in following manuscript

> **The multiscale non-Gaussian statistics of free-running 1000-member general circulation model ensembles** by Man-Yau Chan, Hristo G. Chipilski, Jack Schwartz, Max Albrecht, Aiden Ridgway, and Saurav Dey Shuvo

&nbsp; &nbsp; 

## Overview

The resources provided here support reproducibility of the figures and key diagnostics presented in the manuscript. 
&nbsp; &nbsp; 

## Contents

- `diagnostics_pkl_files/`  
  Directory containing `.pkl` files with ensemble diagnostics, statistics, and other intermediate data products used to create the figures in the manuscript.

- `figures/`  
  Directory where manuscript figures will be outputted

- `SPEEDY_ensemble_data/`
  Directory containing a subset of the SPEEDY ensembles' surface pressure fields

- `manuscript_plot_multiscale_convergence.py` /
  Python script used to produce Figures 1 and 3 of the manuscript.

- `manuscript_plot_psurf2_eigenspace_distribution.py` /
  Python script used to produce Figure 2 of the manuscript.

&nbsp; &nbsp; 


## Requirements

The scripts assume a standard scientific Python environment. Key packages include:

- `numpy`
- `matplotlib`
- `pickle` (standard library)
- `datetime` (standard library)
- `copy` (standard library)
- `os` (standard library)

&nbsp; &nbsp; 



## Additional notes
The `README.md` file you are reading is generated using ChatGPT. The relevant transcript URL is:
> `https://chatgpt.com/share/6890ff2e-3dec-8008-a7fd-8d4656c77c3a`
