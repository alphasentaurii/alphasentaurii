---
layout: post
title:  "AWS Redshift Database Management"
date:   2020-08-10 11:11:11 -1800
categories: datascience
---

Configuring, Managing and Performing Remote SQL Queries on AWS Redshift.

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

# Create SSH Key for Remote Access (optional)

If you don't want to use a password, you can (more securely) access the DB remotely using a key. From the AWS Redshift management console, Go to `Clusters` and click on the cluster we just created. Scroll down to the bottom and copy the SSH public key. On your local machine, create a text file and paste the public key. Save it as something like `redshift_key`. 

In the console, under the Nodes section, copy the public IP address and in the command line/terminal, ssh into the Redshift instance with the DB management user you created above, the public key, and the ip address, for example:

```bash
$ cd ~/.ssh
$ sudo nano redshift_key
# Paste key contents and save
$ ssh -i redshift_key jester@52.54.242.95
```

# Login to AWS Remotely

```bash 
ssh -i key_name username@ipaddress
```

# Import data

Use the kaggle API to download dataset

```bash
$ kaggle competitions download -c trends-assessment-prediction
```
