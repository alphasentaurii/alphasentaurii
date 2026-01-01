---
layout: post
title:  "SPACEKIT Radio: scraping NASA data"
date:   2020-10-10 10:10:10 -1111
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
$ pip install spacekit[x]
```
# Import Packages

```python
import os
from spacekit.extractor.radio import Radio
```

# AWS OpenData Bucket Config (optional)

To acquire data from MAST directly, no additional setup is required (though download speeds may be slower). This demo focuses specifically on downloading from the OpenData s3 bucket.


## Configure AWS access using your account credentials

Before we can fetch data we need to configure our AWS credentials (this demo assumes you have already set up an account). Create a `config` directory and save your credentials in a file called `awscli.ini`. Note this mostly needed when running a notebook - within a python shell Boto3 is typically able to determine creds as soon as you've exported the environment variables.

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
```


# Download data sets via AWS/MAST api

Kepler observed parts of a 10 by 10 degree patch of sky near the constellation of Cygnus for four years (17, 3-month quarters) starting in 2009. The mission downloaded small sections of the sky at a 30-minute (long cadence) and a 1-minute (short cadence) in order to measure the variability of stars and find planets transiting these stars. These data are now available in the public S3 bucket on AWS at s3://stpubdata/kepler/public.

These data are available under the same terms as the public dataset for Hubble and TESS, that is, if you compute against the data from the AWS US-East region, then data access is free.

This script queries MAST for TESS FFI data for a single sector/camera/chip combination and downloads the data from the AWS public dataset rather than from MAST servers.

**Targets with confirmed exoplanets for K2 mission**

```python
os.makedirs('./data/mast', exist_ok=True)
os.chdir('./data/mast')
K2_confirmed_planets = ['K2-1','K2-21','K2-28','K2-39','K2-54','K2-55','K2-57','K2-58','K2-59','K2-60','K2-61','K2-62','K2-63','K2-64','K2-65','K2-66', 'K2-68','K2-70','K2-71','K2-72','K2-73','K2-74','K2-75','K2-76',
'K2-116','K2-167','K2-168','K2-169','K2-170','K2-171','K2-172']
```

```python
radio = Radio(config="enable")
radio.target_list = K2_confirmed_planets
radio.cone_search("0s", "K2", 1800.0, "LLC")
radio.get_object_uris()
radio.s3_download()
```

Download Complete

**Alt: Download larger dataset of all confirmed Kepler planets using `requests` api from NASA**

```python
import requests
r=requests.get("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+pl_name,k2_name+from+k2names&format=json")
results = r.json()

radio.target_list = [r['k2_name'] for r in results]
radio.get_object_uris()
radio.s3_download()
```

In the next several posts, we'll use these datasets to plot light curves and frequency spectrographs then build a convolutional neural network to classify stars that host a transiting exoplanet.
