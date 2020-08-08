---
layout: post
title:  "Stroke Statistics"
date:   2020-08-03 11:11:11 -1800
categories: datascience
---

Predicting stroke outcomes using brain MRI images on AWS Redshift

# Project Goal
Predict age and assessment values from two domains using features derived from brain MRI images as inputs.

# Background: Neuroimaging
Human brain research is among the most complex areas of study for scientists. We know that age and other factors can affect its function and structure, but more research is needed into what specifically occurs within the brain. With much of the research using MRI scans, data scientists are well positioned to support future insights. In particular, neuroimaging specialists look for measurable markers of behavior, health, or disorder to help identify relevant brain regions and their contribution to typical or symptomatic effects.

# Background: TReNDS Competition
The competition(TReNDS) is meant to encourage approaches able to predict age plus additional continuous individual-level assessment values, given multimodal brain features such as 3D functional spatial maps from resting-state functional MRI, static functional network connectivity (FNC) matrices, and source-based morphometry (SBM) loading values from structural MRI. For this task, one of the largest datasets of unbiased multimodal brain imaging features is made available. Given a set of multimodal imaging features, the developed predictors should output age and assessment predictions.

# fMRI Scans
An fMRI scan is a functional magnetic resonance imaging scan that measures and maps the brain’s activity. An fMRI scan uses the same technology as an MRI scan. An MRI is a noninvasive test that uses a strong magnetic field and radio waves to create an image of the brain. The image an MRI scan produces is just of organs/tissue, but an fMRI will produce an image showing the blood flow in the brain. By showing the blood flow it will display which parts of the brain are being stimulated.

# Predictions
We need to predict values for following output variables:

* age
* domain1_var1
* domain1_var2
* domain2_var1
* domain2_var2

# Data Set 
`TReNDS Neuroimaging` : Multiscanner normative age and assessments prediction with brain function, structure, and connectivity.

https://www.kaggle.com/c/trends-assessment-prediction/data

# Background
Models are expected to generalize on data from a different scanner/site (site 2). All subjects from site 2 were assigned to the test set, so their scores are not available. While there are fewer site 2 subjects than site 1 subjects in the test set, the total number of subjects from site 2 will not be revealed until after the end of the competition. To make it more interesting, the IDs of some site 2 subjects have been revealed below. Use this to inform your models about site effects. Site effects are a form of bias. To generalize well, models should learn features that are not related to or driven by site effects.

## Files
The .mat files for this competition can be read in python using h5py, and the .nii file can be read in python using nilearn.

- fMRI_train - a folder containing 53 3D spatial maps for train samples in .mat format
- fMRI_test - a folder containing 53 3D spatial maps for test samples in .mat format
- fnc.csv - static FNC correlation features for both train and test samples
- loading.csv - sMRI SBM loadings for both train and test samples
- train_scores.csv - age and assessment values for train samples
- sample_submission.csv - a sample submission file in the correct format
- reveal_ID_site2.csv - a list of subject IDs whose data was collected with a different scanner than the train samples
- fMRI_mask.nii - a 3D binary spatial map
- ICN_numbers.txt - intrinsic connectivity network numbers for each fMRI spatial map; matches FNC names

# Disclaimer (from Kaggle)

The scores (see train_scores.csv) are not the original age and raw assessment values. They have been transformed and de-identified to help protect subject identity and minimize the risk of unethical usage of the data. Nonetheless, they are directly derived from the original assessment values and, thus, associations with the provided features is equally likely.

Before transformation, the age in the training set is rounded to nearest year for privacy reasons. However, age is not rounded to year (higher precision) in the test set. Thus, heavily overfitting to the training set age will very likely have a negative impact on your submissions.

# How Features Were Obtained
An unbiased strategy was utilized to obtain the provided features. This means that a separate, unrelated large imaging dataset was utilized to learn feature templates. Then, these templates were "projected" onto the original imaging data of each subject used for this competition using spatially constrained independent component analysis (scICA) via group information guided ICA (GIG-ICA).

# 1st set
`Source-based morphometry (SBM) loadings`: These are subject-level weights from a group-level ICA decomposition of gray matter concentration maps from structural MRI (sMRI) scans.

# 2nd set 
`Static functional network connectivity (FNC) matrices`: These are the subject-level cross-correlation values among 53 component timecourses estimated from GIG-ICA of resting state functional MRI (fMRI).

# 3rd set 
`Component spatial maps (SM)`: These are the subject-level 3D images of 53 spatial networks estimated from GIG-ICA of resting state functional MRI (fMRI).

# Optional: Use AWS Redshift for Data Repository
See my blog post on [/blog/datascience/2020/08/08/aws-redshift-configuration.html](configuring and querying AWS Redshift) for housing the data (~160GB) instead of using your local machine.

# Import data

Use the kaggle API to download dataset

```bash
$ kaggle competitions download -c trends-assessment-prediction
```

