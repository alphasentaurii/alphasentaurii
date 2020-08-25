---
layout: post
title:  "Scraping the MAST API via AWS"
date:   2020-06-01 11:11:11 -1111
categories: datascience
---

The vision for `Starskøpe` : cyberoptic artificial telescope is to aggregate large datasets from multiple missions in order to give us a more accurate, more detailed picture of the stars and planets than what we have available to us in the limited view of a single picture from a single telescope at a single point in time.

_This is a continuation of the Starskøpe Project:_
[STARSKØPE I](/datascience/2020/04/01/starskope-cyberoptic-artificial-telescope.html)
[STARSKØPE II](/datascience/2020/05/06/spectrograph-image-classification.html)

## STARSKØPE Phase 3 Objectives

1. Use datasets from the MAST website (via API) to incorporate other calculations of the star's properties as features to be used for classification algorithms. Furthermore, attempt other types of transformations and normalizations on the data before running the model - for instance, apply a Fourier transform.

2. Combine data from multiple campaigns and perhaps even multiple telescopes (for instance, matching sky coordinates and time intervals between K2, Kepler, and TESS for a batch of stars that have overlapping observations - this would be critical for finding transit periods that are longer than the campaigns of a single telecope's observation period).

3. Explore using computer vision on not only the Full Frame images we can collect from telescopes like TESS, but also on spectographs of the flux values themselves. The beauty of machine learning is our ability to rely on the computer to pick up very small nuances in differences that we ourselves cannot see with our own eyes. 
   
4. Explore using autoencoded machine learning algorithms with Restricted Boltzmann Machines - this type of model has proven to be incredibly effective in the image analysis of handwriting as we've seen applied the MNIST dataset - let's find out if the same is true for images of stars, be they the Full Frame Images or spectographs.

---

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

Cload data access is enabled using the enable_cloud_dataset function, which will cause AWS to become the prefered source for data access until it is disabled (disable_cloud_dataset).


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