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


```python
# This script queries MAST for TESS FFI data for a single sector/camera/chip 
# combination and downloads the data from the AWS public dataset rather than 
# from MAST servers.

# Working with http://astroquery.readthedocs.io/en/latest/mast/mast.html
# Make sure you're running the latest version of Astroquery:
# pip install https://github.com/astropy/astroquery/archive/master.zip

from astroquery.mast import Observations
import boto3

# Query for observations in sector 1 (s0001), camera 1, chip 1 (1-1)
obsTable = Observations.query_criteria(obs_id="tess-s0001-1-1")

# Get the products associated with these observations
products = Observations.get_product_list(obsTable)

# Return only the calibrated FFIs (.ffic.fits)
filtered = Observations.filter_products(products, 
                                        productSubGroupDescription="FFIC",
                                        mrp_only=False)

len(filtered)
# > 1282

# Enable 'cloud mode' for module which will return S3-like URLs for FITs files
# e.g. s3://stpubdata/tess/.../tess2018206192942-s0001-1-1-0120-s_ffic.fits
Observations.enable_cloud_dataset()

# Grab the S3 URLs for each of the observations
s3_urls = Observations.get_cloud_uris(filtered)

s3 = boto3.resource('s3')

# Create an authenticated S3 session. Note, download within US-East is free
# e.g. to a node on EC2.
s3_client = boto3.client('s3',
                         aws_access_key_id='YOURAWSACCESSKEY',
                         aws_secret_access_key='YOURSECRETACCESSKEY')

bucket = s3.Bucket('stpubdata')

# Just download a few of the files (remove the [0:3] to download them all)
for url in s3_urls[0:3]:
  # Extract the S3 key from the S3 URL
  fits_s3_key = url.replace("s3://stpubdata/", "")
  root = url.split('/')[-1]
  bucket.download_file(fits_s3_key, root, ExtraArgs={"RequestPayer": "requester"})
  
```