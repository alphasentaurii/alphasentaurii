---
layout: post
title:  "SPACEKIT - Python Library for Astrophysics Machine Learning"
date:   2020-09-09 09:09:09 -1111
categories: datascience, astrophysics
tags: spacekit, machine learning, pypu
---


`spacekit` is a PyPi Machine Learning Utility Package for Astrophysical Data Science.

The original purpose for creating this library was to for perform signal analysis and machine learning classification algorithms on astrophysical (sparse) datasets. The dataset used as an example here is from MAST (Mikulsky Archive for Space Telescopes) accessed via AWS api. We are analyzing time-series light curves (flux signals) of stars from the K2 telescope to identify possible orbiting exoplanets (also known as `threshold crossing events` or TCEs). 

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

- Analyzer: flux-timeseries signal analysis
    - atomic_vector_plotter: Plots scatter and line plots of time series signal values.
    - make_specgram: generate and save spectographs of flux signal frequencies
    - planet_hunter: calculate period, plot folded lightcurve from .fits files


- Transformer: tools for converting and preprocessing signals as numpy arrays
    - hypersonic_pliers: 
    - thermo_fusion_chisel: 
    - babel_fish_dispenser: adds a 1D uniform noise filter using timesteps
    - fast_fourier: fast fourier transform utility function

- Builder: building and fitting convolutional neural networks
    - build_cnn: builds keras 1D CNN architecture
    - fit_cnn: trains keras CNN

- Computer: gets model predictions and evaluates metrics
    - get_preds
    - fnfp
    - keras_history
    - roc_plots
    - compute

## spacekit.Analyzer()
flux-timeseries signal analysis

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
A = Analyzer()
data = './DATA/mast/'
files = os.listdir(data)
f9 =files[9]
A.planet_hunter(f9, fmt='kepler.fits')
```


## spacekit.Transformer()
tools for converting and preprocessing signals as numpy arrays

### hypersonic_pliers

### thermo_fusion_chisel

### babel_fish_dispenser

### fast_fourier

## spacekit.Builder()
building and fitting convolutional neural networks

### build_cnn

### fit_cnn

## spacekit.Computer()
gets model predictions and evaluates metrics

### get_preds

### fnfp

### keras_history

### fusion_matrix

### roc_plots

### compute





# Create AWS secret access key

- Login to the aws console and navigate to IAM. 
- Apply S3 Full Access policy to your user
- Create Access Key - copy and paste the values somewhere (or download the csv) since you won't be able to view them again

In the command line, add your credentials to an /.aws/config file:
```bash
$ mkdir ~/.aws
$ cd ~/.aws
$ touch config
$ nano config
```

Edit the config file and paste in your access key values. **NOTE**:In order to access the MAST data without being charged, you need to use the US-EAST-1 region. 

```bash
#!/bin/sh

[default]
aws_access_key_id=<your secret access key ID>
aws_secret_access_key=<your secret access key>
region=us-east-1
```

# Using Google Colabs

To have AWS cli work in Google Colab, a configuration folder under the path “content/drive/My Drive/” called “config” needs to be created as a .ini file that contains credentials to be stored.

```python
!pip install awscli
!pip install astroquery
```

```python
import pandas as pd
import numpy as np
import os
from astroquery.mast import Observations
from astroquery.mast import Catalogs
import boto3
```

Authorize and mount gdrive

```python
from google.colab import drive
drive.mount('/gdrive',force_remount=True)
```

Enter authorization code and hit enter

_output_: Mounted at /gdrive

Create config directory

```python
%cd '/gdrive/My Drive/'
%mkdir config
%pwd
```

Create the .ini file 

```python
text = '''
[default]
aws_access_key_id = <your access key id> 
aws_secret_access_key = <your secret access key>
region = <your region>
'''
path = "/content/drive/My Drive/config/awscli.ini"
with open(path, 'w') as f:
   f.write(text)
!cat /content/drive/My\ Drive/config/awscli.ini
```

The above script only needs to be run once, since it is equivalent to saving an username and password to a file to be accessed later.

```python
!export AWS_SHARED_CREDENTIALS_FILE=/gdrive/My\ Drive/config/awscli.ini
path = path
os.environ['AWS_SHARED_CREDENTIALS_FILE'] = path
print(os.environ['AWS_SHARED_CREDENTIALS_FILE'])
```
_output_: /gdrive/My Drive/config/awscli.ini

Cloud data access is enabled using the enable_cloud_dataset function, which will cause AWS to become the prefered source for data access until it is disabled (disable_cloud_dataset).


```python
# Getting the cloud URIs
obs_table = Observations.query_criteria(obs_collection=['K2'],
                                        objectname="K2-62",
                                        filters='KEPLER',
                                        provenance_name='K2')
products = Observations.get_product_list(obs_table)
filtered = Observations.filter_products(products,
                                        productSubGroupDescription='LLC')
s3_uris = Observations.get_cloud_uris(filtered)
print(s3_uris)
```

_output_: ['s3://stpubdata/k2/public/lightcurves/c3/206000000/89000/ktwo206089508-c03_llc.fits', 's3://stpubdata/k2/public/lightcurves/c3/206000000/92000/ktwo206092110-c03_llc.fits', 's3://stpubdata/k2/public/lightcurves/c3/206000000/92000/ktwo206092615-c03_llc.fits', 's3://stpubdata/k2/public/lightcurves/c3/206000000/93000/ktwo206093036-c03_llc.fits', 's3://stpubdata/k2/public/lightcurves/c3/206000000/93000/ktwo206093540-c03_llc.fits', 's3://stpubdata/k2/public/lightcurves/c3/206000000/94000/ktwo206094039-c03_llc.fits', 's3://stpubdata/k2/public/lightcurves/c3/206000000/94000/ktwo206094098-c03_llc.fits', 's3://stpubdata/k2/public/lightcurves/c3/206000000/94000/ktwo206094342-c03_llc.fits', 's3://stpubdata/k2/public/lightcurves/c3/206000000/94000/ktwo206094605-c03_llc.fits', 's3://stpubdata/k2/public/lightcurves/c3/206000000/95000/ktwo206095133-c03_llc.fits', 's3://stpubdata/k2/public/lightcurves/c3/206000000/96000/ktwo206096022-c03_llc.fits', 's3://stpubdata/k2/public/lightcurves/c3/206000000/96000/ktwo206096602-c03_llc.fits', 's3://stpubdata/k2/public/lightcurves/c3/206000000/96000/ktwo206096692-c03_llc.fits', 's3://stpubdata/k2/public/lightcurves/c3/206000000/97000/ktwo206097453-c03_llc.fits', 's3://stpubdata/k2/public/lightcurves/c3/206000000/98000/ktwo206098619-c03_llc.fits', 's3://stpubdata/k2/public/lightcurves/c3/206000000/98000/ktwo206098990-c03_llc.fits', 's3://stpubdata/k2/public/lightcurves/c3/206000000/99000/ktwo206099456-c03_llc.fits', 's3://stpubdata/k2/public/lightcurves/c3/206000000/99000/ktwo206099582-c03_llc.fits', 's3://stpubdata/k2/public/lightcurves/c3/206000000/99000/ktwo206099965-c03_llc.fits', 's3://stpubdata/k2/public/lightcurves/c3/206100000/00000/ktwo206100060-c03_llc.fits', 's3://stpubdata/k2/public/lightcurves/c3/206100000/02000/ktwo206102898-c03_llc.fits', 's3://stpubdata/k2/public/lightcurves/c3/206100000/03000/ktwo206103033-c03_llc.fits', 's3://stpubdata/k2/public/lightcurves/c3/212200000/35000/ktwo212235329-c03_llc.fits']


# download the FITS files

```python
for url in s3_urls:
  # Extract the S3 key from the S3 URL
  fits_s3_key = url.replace("s3://stpubdata/", "")
  root = url.split('/')[-1]
  bucket.download_file(fits_s3_key, root, ExtraArgs={"RequestPayer": "requester"})
```

# Analyze FITS files (Light Curves)

```python
import matplotlib.pyplot as plt
%matplotlib inline
!pip install astropy

import tarfile
from astropy.utils.data import download_file
url = 'http://data.astropy.org/tutorials/UVES/data_UVES.tar.gz'
f = tarfile.open(download_file(url, cache=True), mode='r|*')
working_dir_path = '.'  # CHANGE TO WHEREVER YOU WANT THE DATA TO BE EXTRACTED
f.extractall(path=working_dir_path)
```

You should now have all the FITS files saved in your google drive folder.

---

