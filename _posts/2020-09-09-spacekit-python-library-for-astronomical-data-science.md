---
layout: post
title:  "SPACEKIT: Machine Learning for Astrophysics"
date:   2020-09-09 09:09:09 -1111
categories: datascience
tags: spacekit machine-learning pypi astrophysics
author: Ru Keïn
---

# `spacekit` is a PyPi Machine Learning Utility Package for Astrophysical Data Science.

This library is for performing signal analysis and machine learning on astrophysical datasets. The dataset used as an example here is from MAST (Mikulsky Archive for Space Telescopes) accessed via AWS api. We are analyzing time-series light curves (flux signals) of stars from the K2 telescope to identify possible orbiting exoplanets (also known as `threshold crossing events` or TCEs). 

This post shows how to install spacekit locally and presents an overview of the library. The next few posts include walkthroughs and demos for each specific class and their respective class methods.

# Source Code

```bash
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

# Install spacekit via `pip`

```bash
$ pip install spacekit
```

# [spacekit.radio](/datascience/2020/10/10/spacekit-radio-scraping-nasa-api.html)
Scrape Mikulsky archives (MAST) for downloading NASA space telescope datasets
- mast_aws: fetch data hosted on AWS using the MAST api 

# [spacekit.analyzer]('/datascience/2020/11/11/spacekit-analyzer-plotting-light-curves.html')
flux-timeseries signal analysis

[spacekit.analyzer (part 1): plotting light curves]('/datascience/2020/11/11/spacekit-analyzer-plotting-light-curves.html')
[spacekit.analyzer (part 2): frequency spectrographs]('/datascience/2020/12/12/spacekit-analyzer-frequency-spectrographs.html')

- atomic_vector_plotter: Plots scatter and line plots of time series signal values.
- planet_hunter: calculate period, plot folded lightcurve from .fits files
- make_specgram: generate and save spectographs of flux signal frequencies

## [spacekit.transformer](/datascience/2021/01/01-spacekit-transformer-signal-processing-and-analysis.html)
tools for converting and preprocessing signals as numpy arrays

- hypersonic_pliers: load datasets from file and extract into 1D arrays 
- thermo_fusion_chisel: scale multiple arrays to zero mean and unit variance.
- babel_fish_dispenser: adds a 1D uniform noise filter using timesteps
- fast_fourier: fast fourier transform utility function

## spacekit.builder
building and fitting convolutional neural networks

- build_cnn: builds and compiles linear CNN using Keras
- batch_maker: pass equal number of class samples rotating randomly
- fit_cnn: trains keras CNN

## spacekit.computer
gets model predictions and evaluates metrics

- get_preds: generate model predictions
- fnfp: count of false negative and false positive predictions
- keras_history: keras history plots (accuracy and loss)
- fusion_matrix: customized multi-class confusion matrix
- roc_plots: receiver operator characteristic (ROC) plot
- compute: generates all of the above in one shot