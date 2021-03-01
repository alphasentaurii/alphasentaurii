---
layout: post
title:  "AWS Redshift Database Management"
date:   2019-11-11 11:11:11 -1800
categories: programming
tags: aws, redshift, sql
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

- Step 1: Retrieve the cluster public key and cluster node IP addresses

If you don't want to use a password, you can (more securely) access the DB remotely using a key. From the AWS Redshift management console, Go to `Clusters` and click on the cluster we just created. Scroll down to the bottom and copy the SSH public key. On your local machine, create a text file and paste the public key. Save it as something like `redshift_key`. 

- Step 2: Add the Amazon Redshift cluster public key to the host's authorized keys file

In the console, under the Nodes section, copy the public IP address and in the command line/terminal, ssh into the Redshift instance with the DB management user you created above, the public key, and the ip address, for example:

```bash
$ cd ~/.ssh
$ sudo nano redshift_key
# Paste key contents and save

# change permissions
$ chmod 0400 redshift_key

$ ssh -L localhost:8888:localhost:8888 -i redshift_key ec2-user@ec2-3-236-65-85.compute-1.amazonaws.com
```

Add to config file
```bash
$ sudo nano config #if using a Mac
#
Host 52.54.242.95
   User jester
   IdentityFile ~/.ssh/redshift_key
```

You add the Amazon Redshift cluster public key to the host's authorized keys file so that the host will recognize the Amazon Redshift cluster and accept the SSH connection.

# Modify Security Groups

For Amazon EC2 , modify the instance's security groups to add ingress rules to accept the Amazon Redshift IP addresses. For other hosts, modify the firewall so that your Amazon Redshift nodes are able to establish SSH connections to the remote host.

# Load from AWS S3 Bucket

Loading data into your Amazon Redshift database tables from data files in an Amazon S3 bucket

1. Create an Amazon S3 bucket and then upload the data files to the bucket.

2. Launch an Amazon Redshift cluster and create database tables.

3. Use COPY commands to load the tables from the data files on Amazon S3.


# Run the COPY command to load the data

From an Amazon Redshift database, run the COPY command to load the data into an Amazon Redshift table.

# Login to AWS Remotely

Replace "jester" with your Redshift Master username and the ip address with the public IP of your EC2 node.
```bash 
$ ssh -i redshift_key jester@52.54.242.95
```

# Import data

Use API to download dataset

```bash
$ kaggle competitions download -c trends-assessment-prediction
```
