---
layout: post
title:  "Stroke Statistics"
date:   2020-08-03 11:11:11 -1800
categories: datascience
---

Predict age and assessment values from two domains using features derived from brain MRI images as inputs.

# Abstract
Human brain research is among the most complex areas of study for scientists. We know that age and other factors can affect its function and structure, but more research is needed into what specifically occurs within the brain. With much of the research using MRI scans, data scientists are well positioned to support future insights. In particular, neuroimaging specialists look for measurable markers of behavior, health, or disorder to help identify relevant brain regions and their contribution to typical or symptomatic effects.

# Methods
Approaches to predicting age plus additional continuous individual-level assessment values, given multimodal brain features such as:
- 3D functional spatial maps from resting-state functional MRI
- static functional network connectivity (FNC) matrices
- source-based morphometry (SBM)

...loading values from structural MRI. 

This project uses one of the largest datasets of unbiased multimodal brain imaging features available. 

Outcome: Given a set of multimodal imaging features, the developed predictors should output age and assessment predictions.

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

Source:
https://www.kaggle.com/c/trends-assessment-prediction/data

# Model
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

# Scores

The scores (see train_scores.csv) are not the original age and raw assessment values. They have been transformed and de-identified to help protect subject identity and minimize the risk of unethical usage of the data. Nonetheless, they are directly derived from the original assessment values and, thus, associations with the provided features is equally likely.

Before transformation, the age in the training set is rounded to nearest year for privacy reasons. However, age is not rounded to year (higher precision) in the test set. Thus, heavily overfitting to the training set age will very likely have a negative impact on your submissions.

# Features
An unbiased strategy was utilized to obtain the provided features. This means that a separate, unrelated large imaging dataset was utilized to learn feature templates. Then, these templates were "projected" onto the original imaging data of each subject used for this competition using spatially constrained independent component analysis (scICA) via group information guided ICA (GIG-ICA).

## 1st set
`Source-based morphometry (SBM) loadings`: These are subject-level weights from a group-level ICA decomposition of gray matter concentration maps from structural MRI (sMRI) scans.

## 2nd set 
`Static functional network connectivity (FNC) matrices`: These are the subject-level cross-correlation values among 53 component timecourses estimated from GIG-ICA of resting state functional MRI (fMRI).

## 3rd set 
`Component spatial maps (SM)`: These are the subject-level 3D images of 53 spatial networks estimated from GIG-ICA of resting state functional MRI (fMRI).

---

# CODE 

```python
import os
import random
import seaborn as sns
import cv2

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import PIL

import plotly.graph_objs as go
from IPython.display import Image, display
import joypy
import warnings
warnings.filterwarnings("ignore")
import nilearn as nl
import nilearn.plotting as nlpl
import nibabel as nib
import h5py
```

```python
os.listdir('/kaggle/input/trends-assessment-prediction/')


['train_scores.csv',
 'fMRI_test',
 'fMRI_mask.nii',
 'reveal_ID_site2.csv',
 'fnc.csv',
 'fMRI_train',
 'loading.csv',
 'ICN_numbers.csv',
 'sample_submission.csv']

```

```python
BASE_PATH = '../input/trends-assessment-prediction'

# image and mask directories
train_data_dir = f'{BASE_PATH}/fMRI_train'
test_data_dir = f'{BASE_PATH}/fMRI_test'

# load dataframes
df_loading = pd.read_csv(f'{BASE_PATH}/loading.csv')
df_train = pd.read_csv(f'{BASE_PATH}/train_scores.csv')
df_sub = pd.read_csv(f'{BASE_PATH}/sample_submission.csv')

```

# Preprocessing

Training data

```python
display(train_data.head())
print("Shape of train_data :", train_data.shape)

    Id	    age	     domain1_var1 domain1_var2 domain2_var1 domain2_var2
0	10001	57.436077	30.571975	62.553736	53.325130	51.427998
1	10002	59.580851	50.969456	67.470628	60.651856	58.311361
2	10004	71.413018	53.152498	58.012103	52.418389	62.536641
3	10005	66.532630	NaN	        NaN	        52.108977	69.993075
4	10007	38.617381	49.197021	65.674285	40.151376	34.096421

Shape of train_data : (5877, 6)

```

Test Data

```python
display(loading_data.head())
print("Shape of loading_data :", loading_data.shape)

Id	IC_01	IC_07	IC_05	IC_16	IC_26	IC_06	IC_10	IC_09	IC_18	...	IC_08	IC_03	IC_21	IC_28	IC_11	IC_20	IC_30	IC_22	IC_29	IC_14
0	10001	0.006070	0.014466	0.004136	0.000658	-0.002742	0.005033	0.016720	0.003484	0.001797	...	0.018246	0.023711	0.009177	-0.013929	0.030696	0.010496	0.002892	-0.023235	0.022177	0.017192
1	10002	0.009087	0.009291	0.007049	-0.002076	-0.002227	0.004605	0.012277	0.002946	0.004086	...	0.014635	0.022556	0.012004	-0.011814	0.022479	0.005739	0.002880	-0.016609	0.025543	0.014524
2	10003	0.008151	0.014684	0.010444	-0.005293	-0.002913	0.015042	0.017745	0.003930	-0.008021	...	0.019565	0.030616	0.018184	-0.010469	0.029799	0.015435	0.005211	-0.028882	0.031427	0.018164
3	10004	0.004675	0.000957	0.006154	-0.000429	-0.001222	0.011755	0.013010	0.000193	0.008075	...	0.002658	0.022266	0.005956	-0.010595	0.024078	-0.000319	0.005866	-0.015182	0.024476	0.014760
4	10005	-0.000398	0.006878	0.009051	0.000369	0.000336	0.010679	0.010352	0.003637	0.004180	...	0.009702	0.017257	0.005454	-0.008591	0.019416	0.000786	0.002692	-0.019814	0.017105	0.013316
5 rows × 27 columns

Shape of loading_data : (11754, 27)
```

# Check Nulls

```python
# checking missing data
total = train_data.isnull().sum().sort_values(ascending = False)
percent = (train_data.isnull().sum()/train_data.isnull().count()*100).sort_values(ascending = False)
missing_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_train_data.head()

	Total	Percent
domain1_var2	438	7.452782
domain1_var1	438	7.452782
domain2_var2	39	0.663604
domain2_var1	39	0.663604
age	0	0.000000

```
# Percentage Missing Data

```python
total = loading_data.isnull().sum().sort_values(ascending = False)
percent = (loading_data.isnull().sum()/loading_data.isnull().count()*100).sort_values(ascending = False)
missing_loading_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_loading_data.head()
```

# EDA

```python
def plot_bar(df, feature, title='', show_percent = False, size=2):
    f, ax = plt.subplots(1,1, figsize=(4*size,3*size))
    total = float(len(df))
    sns.barplot(np.round(df[feature].value_counts().index).astype(int), df[feature].value_counts().values, alpha=0.8, palette='bright')

    plt.title(title)
    if show_percent:
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x()+p.get_width()/2.,
                    height + 3,
                    '{:1.2f}%'.format(100*height/total),
                    ha="center", rotation=45) 
    plt.xlabel(feature, fontsize=12, )
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xticks(rotation=90)
    plt.show()


plot_bar(train_data, 'age', 'age count and %age plot', show_percent=True, size=4)   
```

