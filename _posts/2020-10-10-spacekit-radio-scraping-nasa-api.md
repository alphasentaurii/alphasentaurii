---
layout: post
title:  "SPACEKIT Radio: scraping NASA data"
date:   2020-10-10 10:10:10 -1800
categories: datascience
tags: spacekit astrophysics aws s3 nasa api
author: Ru Keïn
---

# spacekit.radio
This module is used to access the STScI public dataset [astroquery.mast.cloud] hosted in S3 on AWS. In this post, I'll show you how to scrape and download NASA space telescope datasets that can be used in astronomical machine learning projects (or any astronomy-based analysis and programming). For this demonstration we'll call the API to acquire FITS files containing the time-validated light curves of stars with confirmed exoplanets. The datasets all come from the K2 space telescope (Kepler phase 2).

# Prerequisites
Creation of a virtual-env is recommended.

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
$ pip install spacekit
```

```python
import pandas as pd
import numpy as np
import os
from astroquery.mast import Observations, Catalogs
import boto3
from spacekit import radio
```

# spacekit.radio.mast_aws(target_list)
This function fetches data hosted on AWS (via Space Telescope Science Institute using their API for the Mikulsky Archives (MAST).


```python
# function for downloading data from MAST s3 bucket on AWS
def mast_aws(target_list):
    import boto3
    from astroquery.mast import Observations
    from astroquery.mast import Catalogs
    # configure aws settings
    region = 'us-east-1'
    s3 = boto3.resource('s3', region_name=region)
    bucket = s3.Bucket('stpubdata')
    location = {'LocationConstraint': region}
    Observations.enable_cloud_dataset(provider='AWS', profile='default') # make AWS preferred data source
    
    for target in target_list:
    #Do a cone search and find the K2 long cadence data for target
        obs = Observations.query_object(target,radius="0s")
        want = (obs['obs_collection'] == "K2") & (obs['t_exptime'] ==1800.0)
        data_prod = Observations.get_product_list(obs[want])
        filt_prod = Observations.filter_products(data_prod, productSubGroupDescription="LLC")
        s3_uris = Observations.get_cloud_uris(filt_prod)
        for url in s3_uris:
        # Extract the S3 key from the S3 URL
            fits_s3_key = url.replace("s3://stpubdata/", "")
            root = url.split('/')[-1]
            bucket.download_file(fits_s3_key, root, ExtraArgs={"RequestPayer": "requester"})
    Observations.disable_cloud_dataset()
    return print('Download Complete')
```


## Configure AWS access using your account credentials
Before we can fetch data we need to configure our AWS credentials (this demo assumes you have already set up an account) and configure Boto3. Create a `config` directory and save your credentials in a file called `awscli.ini`.  

```python
os.makedirs('config', exist_ok=True)
text = '''
[default]
aws_access_key_id = <access_id>
aws_secret_access_key = <access_key>
aws_session_token= <token>
'''
path = "./config/awscli.ini"
with open(path, 'w') as f:
    f.write(text)
```

### Set the credentials via config file
Now that we have our credentials stored in a file locally, we can set the path as an environment variable and call it from within the notebook (Jupyter or Google Colab). 

```python
!export AWS_SHARED_CREDENTIALS_FILE=./config/awscli.ini
path = path
os.environ['AWS_SHARED_CREDENTIALS_FILE'] = path
print(os.environ['AWS_SHARED_CREDENTIALS_FILE'])
```

### Setup Boto3 configuration
Make sure you set your region to `us-east-1`; at least for now, accessing the data from within this region is free of cost.  
For more info on how to configure Boto3, check out the guide here:
https://boto3.readthedocs.io/en/latest/guide/configuration.html [astroquery.mast.cloud]

```python
region = 'us-east-1'
s3 = boto3.resource('s3', region_name=region)
bucket = s3.Bucket('stpubdata')
location = {'LocationConstraint': region}
```

# Download data sets via AWS/MAST api
Download data from s3 bucket on AWS using the `spacekit.radio` class method: `mast_aws`.

**Notes:**

Kepler observed parts of a 10 by 10 degree patch of sky near the constellation of Cygnus for four years (17, 3-month quarters) starting in 2009. The mission downloaded small sections of the sky at a 30-minute (long cadence) and a 1-minute (short cadence) in order to measure the variability of stars and find planets transiting these stars. These data are now available in the public s3://stpubdata/kepler/public S3 bucket on AWS.

These data are available under the same terms as the public dataset for Hubble and TESS, that is, if you compute against the data from the AWS US-East region, then data access is free.

This script queries MAST for TESS FFI data for a single sector/camera/chip combination and downloads the data from the AWS public dataset rather than from MAST servers.q

**Targets with confirmed exoplanets for K2 mission**
```python
os.makedirs('./data/mast', exist_ok=True)
os.chdir('./data/mast')
K2_confirmed_planets = ['K2-1','K2-21','K2-28','K2-39','K2-54','K2-55','K2-57','K2-58','K2-59','K2-60','K2-61','K2-62','K2-63','K2-64','K2-65','K2-66', 'K2-68','K2-70','K2-71','K2-72','K2-73','K2-74','K2-75','K2-76',
'K2-116','K2-167','K2-168','K2-169','K2-170','K2-171','K2-172']
```

```python
from spacekit.radio import Radio
radio = Radio()
radio.mast_aws(K2_confirmed_planets)
```

Download Complete

**Alt: Download larger dataset of all confirmed Kepler planets using `requests` api from NASA**

```python
import requests
resp = requests.get('https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=exoplanets&select=pl_hostname,ra,dec&where=pl_hostname like K2&format=json')

r=requests.get("https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=exoplanets&format=json&select=pl_hostname&where=pl_hostname like '%K2%'")
results = r.json()

targets_df = pd.DataFrame.from_dict(results)

k2_targets = list(targets_df['pl_hostname'].unique())

radio.mast_aws(k2_targets)

```

```python
MAST = './data/mast'
len(os.listdir(MAST))
```

`348`

```python
os.listdir(MAST)[9]
```

`'ktwo246067459-c12_llc.fits'`

In the next several posts, we'll use these datasets to plot light curves and frequency spectrographs then build a convolutional neural network to classify stars that host a transiting exoplanet.


# NEXT

[spacekit.analyzer (part 1): plotting light curves]('/datascience/2020/11/11/spacekit-analyzer-plotting-light-curves.html')

