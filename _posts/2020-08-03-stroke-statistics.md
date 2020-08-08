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

Amazon offers 2 months free to AWS users (assuming this is your first time using it).  Login to the console and go to the AWS Redshift Console. Click `Create Cluster` then select `free trial` and a `DC2.Large` cluster. This will give you up to 160GB SSD. 

Database Configurations

* Database name: dev
* Database port: 5439
* Master user name: jester
* Master user password: [password]

IAM Redshift Access Configuration (optional)

- Open AWS Console in a new tab 
- Go to IAM 
- GO to Roles
- Select `Create Role`
- Select `AWS Service` 
- Click `Redshift` and select `Redshift - Customizable`
- Attach policy: `RedshiftFullAccess`
- Enter a name: [redshift]
- Hit Create
- Example: `arn:aws:iam::***********:role/redshift`
- Hit Refresh button next to Available IAM role
- Select Role you just created `redshift`




