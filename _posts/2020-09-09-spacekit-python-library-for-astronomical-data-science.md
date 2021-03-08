---
layout: post
title:  "SPACEKIT: Machine Learning for Astrophysics"
date:   2020-09-09 09:09:09 -1111
categories: datascience, astrophysics
tags: spacekit, machine learning, pypu
---

`spacekit` is a PyPi Machine Learning Utility Package for Astrophysical Data Science.

This library is for performing signal analysis and machine learning on astrophysical datasets. The dataset used as an example here is from MAST (Mikulsky Archive for Space Telescopes) accessed via AWS api. We are analyzing time-series light curves (flux signals) of stars from the K2 telescope to identify possible orbiting exoplanets (also known as `threshold crossing events` or TCEs). 

This post shows how to install spacekit locally and presents an overview of the library. Posts following this one walk you through a demo of each specific class and class methods. 

# Prerequisites

- Creation of a virtual-env is recommended.
- an AWS account (use `us-east-1` region)
- awscli
- astroquery
- boto3
- numpy
- pandas

# Install Dependencies

```bash
$ pip install awscli
$ pip install astroquery
```

```python
import pandas as pd
import numpy as np
import os
from astroquery.mast import Observations
from astroquery.mast import Catalogs
import boto3
```

# Source Code

```python
spacekit
└── spacekit_pkg
    └── __init__.py
    └── analyzer.py
    └── builder.py
    └── computer.py
    └── radio.py
    └── transformer.py
└── setup.py
└── tests
└── LICENSE
└── README.md
```

## Install spacekit via `pip`

```bash
$ pip install spacekit
```

## spacekit.Radio()
Scrape Mikulsky archives (MAST) for downloading NASA space telescope datasets
- mast_aws: fetch data hosted on AWS using the MAST api 


## spacekit.Analyzer()
flux-timeseries signal analysis

- atomic_vector_plotter: Plots scatter and line plots of time series signal values.
- make_specgram: generate and save spectographs of flux signal frequencies
- planet_hunter: calculate period, plot folded lightcurve from .fits files

### atomic_vector_plotter
Plots scatter and line plots of time series signal values.

```python
from spacekit import analyzer
signal = array([  93.85,   83.81,   20.1 ,  -26.98,  -39.56, -124.71, -135.18,
        -96.27,  -79.89, -160.17, -207.47, -154.88, -173.71, -146.56,
       -120.26, -102.85,  -98.71,  -48.42,  -86.57,   -0.84,  -25.85,
        -67.39,  -36.55,  -87.01,  -97.72, -131.59, -134.8 , -186.97,
       -244.32, -225.76, -229.6 , -253.48, -145.74, -145.74,   30.47,
       -173.39, -187.56, -192.88, -182.76, -195.99, -317.51, -167.69,
        -56.86,    7.56,   37.4 ,  -81.13,  -20.1 ,  -30.34, -320.48,
       -320.48, -287.72, -351.25,  -70.07, -194.34, -106.47,  -14.8 ,
         63.13,  130.03,   76.43,  131.9 , -193.16, -193.16,  -89.26,
        -17.56,  -17.31,  125.62,   68.87,  100.01,   -9.6 ,  -25.39,
        -16.51,  -78.07, -102.15, -102.15,   25.13,   48.57,   92.54,
         39.32,   61.42,    5.08,  -39.54])
A = Analyzer()
A.atomic_vector_plotter(signal)
```

### make_specgram
generate and save spectographs of flux signal frequencies

```python
A = Analyzer()
spec = A.make_specgram(signal)
```

### planet_hunter
calculates period and plots folded light curve from single or multiple .fits files

```python
from spacekit.analyzer import Analyzer
analyzer = Analyzer()
files = os.listdir('./data/mast')
fits_files = files[0]
analyzer.planet_hunter(fits_file, fmt='kepler.fits')
```

## spacekit.Transformer()
tools for converting and preprocessing signals as numpy arrays

- hypersonic_pliers: 
- thermo_fusion_chisel: 
- babel_fish_dispenser: adds a 1D uniform noise filter using timesteps
- fast_fourier: fast fourier transform utility function

## spacekit.Builder()
building and fitting convolutional neural networks

- build_cnn: builds keras 1D CNN architecture
- fit_cnn: trains keras CNN


## spacekit.Computer()
gets model predictions and evaluates metrics

- get_preds
- fnfp
- keras_history
- fusion_matrix
- roc_plots
- compute