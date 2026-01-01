---
layout: post
title:  "SPACEKIT: Machine Learning for Astrophysics"
date:   2020-09-09 09:09:09 -1111
categories: datascience
tags: spacekit machine-learning pypi astrophysics
author: Ru Keïn
---

# Overview

`spacekit` is a python library designed to do the heavy lifting of machine learning in astronomy-related applications. The modules contained in this package can be used to assist and streamline each step of a typical data science project. The library is especially useful for performing signal analysis and machine learning on astronomical datasets. This post shows how to install spacekit locally and presents an overview of the library. The next few posts include walkthroughs and demos including how to acquire and preprocess datasets from MAST (Mikulsky Archive for Space Telescopes) via AWS, analyze time-series light curves (flux signals) of stars from the K2 telescope to identify possible orbiting exoplanets (also known as `threshold crossing events` or TCEs), and more.


## References:

- [Documentation](https://www.spacekit.org)
- [Source Code](https://github.com/spacetelescope/spacekit)



## Ingest/Extract 

Import large datasets from a variety of file formats .csv, .hdf5, .fits, .json, .png .asdf

## Scrub/Preprocess 

Scrub and preprocess raw data to prepare it for use in a machine learning model

## Modeling 

Build, train and deploy custom machine learning models using classification, logistic regression estimation, computer vision and more

## Analysis

Evaluate model performance and do exploratory data analysis (EDA) using interactive graphs and visualizations

## Visualize

Deploy a web-based custom dashboard for your models and datasets via docker, a great way to summarize and share comparative model evaluations and data analysis visuals with others


# Applications

The `Skøpes` module includes real-world machine learning applications used by the Hubble and James Webb Space Telescopes in data calibration pipelines. These mini-applications are an orchestration of functions and classes from other spacekit modules to run real-time, automated analysis, training, and inference on a local server as well as in the cloud (AWS).


# Install spacekit via `pip`

```bash
$ pip install spacekit[x]
```

# spacekit.radio

[Fetch data hosted on AWS using the MAST api](/datascience/2020/10/10/spacekit-radio-scraping-nasa-api.html)

How to scrape MAST archives and download NASA space telescope datasets


# spacekit.analyzer

How to perform flux-timeseries signal analysis with light curves

[spacekit.analyzer (part 1): plotting light curves](/datascience/2020/11/11/spacekit-analyzer-plotting-light-curves.html)
[spacekit.analyzer (part 2): frequency spectrographs](/datascience/2020/12/12/spacekit-analyzer-frequency-spectrographs.html)

- atomic_vector_plotter: Plots scatter and line plots of time series signal values.
- planet_hunter: calculate period, plot folded lightcurve from .fits files
- make_specgram: generate and save spectographs of flux signal frequencies

## spacekit.transformer

[Exploring tools for converting and preprocessing signals as numpy arrays](/datascience/2021/01/01/spacekit-transformer-signal-processing-and-analysis.html)

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
