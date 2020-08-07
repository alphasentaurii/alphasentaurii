---
layout: post
title:  "Stroke Statistics"
date:   2020-08-03 11:11:11 -1800
categories: datascience
---

Predicting stroke outcomes using brain MRI images on AWS Redshift

https://www.kaggle.com/c/trends-assessment-prediction/data

# Import data

Use the kaggle API to download dataset

```bash
$ kaggle competitions download -c trends-assessment-prediction
```

# AWS Redshift

Amazon offers 2 months free to AWS users (assuming this is your first time using it).  Login to the console and go to the AWS Redshift Console. Click `Create Cluster` then select the free tier and a DC2.Large cluster. This will give you up to 160GB SSD. 