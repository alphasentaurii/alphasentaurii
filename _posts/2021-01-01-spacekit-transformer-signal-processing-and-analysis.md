---
layout: post
title:  "SPACEKIT Transformer: signal processing"
date:   2021-01-01 01:01:01 -1111
categories: datascience
tags: spacekit astrophysics
author: Ru Keïn
---


# spacekit.transformer
tools for converting and preprocessing timeseries signals for ML

- hypersonic_pliers: load datasets from file and extract into 1-dimensional numpy arrays
- thermo_fusion_chisel: Scales each array of a matrix to zero mean and unit variance.
- babel_fish_dispenser: adds a 1D uniform noise filter using timesteps
- fast_fourier: fast fourier transform utility function

### hypersonic_pliers
load datasets from file and extract into 1-dimensional numpy arrays

### thermo_fusion_chisel
Scales each array of a matrix to zero mean and unit variance.

### babel_fish_dispenser
Adds an input corresponding to the running average over a set number of time steps. This helps the neural network to ignore high frequency noise by passing in a uniform 1-D filter and stacking the arrays. 

### fast_fourier
takes in array and rotates #bins to the left as a fourier transform returns vector of length equal to input array

```python
                       
           /\    _       _                           _                      *  
/\_/\_____/  \__| |_____| |_________________________| |___________________*___
[===]    / /\ \ | |  _  |  _  | _  \/ __/ -__|  \| \_  _/ _  \ \_/ | * _/| | |
 \./    /_/  \_\|_|  ___|_| |_|__/\_\ \ \____|_|\__| \__/__/\_\___/|_|\_\|_|_|
                  | /             |___/        
                  |/   
```