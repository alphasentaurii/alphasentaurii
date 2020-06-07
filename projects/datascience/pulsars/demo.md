---
layout: page
title: Detecting Dead Stars Project Demo
---

This is a `supervised machine learning feature classification project` that uses `Decision Trees and XGBoost` to `predict and classify signals as either a pulsar or radio frequency interference (noise)`.

<iframe src="https://player.vimeo.com/video/389320953" width="640" height="480" frameborder="0" allow="autoplay; fullscreen" allowfullscreen></iframe>

## HTRU2

HTRU2 is a data set which describes **a sample of pulsar candidates collected during the High Time Resolution Universe Survey.**


<div style="width:400px">
<img src="http://hakkeray.com/assets/images/pulsars/output_93_1.png"></div>

## Pulsars

Pulsars are a rare type of Neutron star that produce radio emission detectable here on Earth. They are of considerable scientific interest as probes of space-time, the inter-stellar medium, and states of matter.

## What happens when they rotate?

Glad you asked. As pulsars rotate, their emission beams sweep across the sky which produces a detectable pattern of broadband radio emission when crossing our line of sight. As pulsars rotate rapidly, this pattern repeats periodically. Thus pulsar search involves looking for periodic radio signals with large radio telescopes.

## So how do we detect pulsars?

Each pulsar produces a slightly different emission pattern, which varies slightly with each rotation. Detection of a potential signal is known as a 'candidate', which is averaged over many rotations of the pulsar, as determined by the length of an observation. 

## Sounds easy enough

The problem is that, in the absence of additional info, each candidate could potentially describe a real pulsar. **However in practice almost all detections are caused by radio frequency interference (RFI) and noise, making legitimate signals hard to find.** Thus, legitimate pulsar examples are a minority positive class, and spurious examples the majority negative class.

<div style="width:400px">
<img src="http://hakkeray.com/assets/images/pulsars/output_20_2.png"></div>

## The Dataset

The data set shared here contains **16,259 spurious examples caused by RFI/noise**, and **1,639 real pulsar examples**. Each row lists the variables first, and the class label is the final entry. The class labels used are 0 (negative) and 1 (positive).

## Features (variables)

Each candidate is described by 8 continuous variables, and a single class variable. The first four are simple statistics obtained from the integrated pulse profile (folded profile). This is an array of continuous variables that describe a longitude-resolved version of the signal that has been averaged in both time and frequency. The remaining four variables are similarly obtained from the DM-SNR curve.

    * Mean of the integrated profile.
    * Standard deviation of the integrated profile.
    * Excess kurtosis of the integrated profile.
    * Skewness of the integrated profile.
    * Mean of the DM-SNR curve.
    * Standard deviation of the DM-SNR curve.
    * Excess kurtosis of the DM-SNR curve.
    * Skewness of the DM-SNR curve.
    * Class

HTRU 2 Summary:

    * 17,898 total examples
            * 1,639 positive examples
            * 16,259 negative examples

<div style="width:400px">
<img src="http://hakkeray.com/assets/images/pulsars/output_91_1.png"></div>


Kurtosis Integrated Profile ('KURTOSIS_IP') is by far the most important classifying feature when it comes to identifying Pulsars. Let's double check the other metrics with our scaled/transformed data:

## Normalized confusion matrix

<div style="width:400px">
<img src="http://hakkeray.com/assets/images/pulsars/output_117_1.png">
</div>


## Confusion matrix, without normalization

<div style="width:400px">
<img src="http://hakkeray.com/assets/images/pulsars/output_118_1.png"></div>

# CONCLUSION

I began analysis with a pipeline to determine the most accurate models for predicting a pulsar. After performing Standard Scaling on the dataset, I split the dataset into train-test prediction models for Logistic Regression, Support Vector Machines, Decision Trees and XG Boost. All were fairly accurate, with Decision Trees and XG Boost topping the list for accuracy scores.

I then proceeded with a Decision Tree classifier with balanced class weights, which did fairly well, scoring 96% accuracy. However, because of the imbalanced classes, the F1 score is our most important validator for model accuracy, and the Decision Tree classifier scored 82%.

Moving on to XGBoost, the model scored 98% accuracy with an 89% F1 score. The model successfully identify 466 pulsars, missing only 78 which it mistakenly identified as noise.

# RECOMMENDATIONS

     * Focus on Kurtosis Integrated Profile
 
     * Focus on Standard Deviation DM-NSR Curve
 
     * Validate model predictions with analysis of other celestial objects 
     producing cosmic rays to see if they show the same attributes.

# FUTURE WORK

1. Improving the model, trying other ways of scaling, balancing class weights.

2. Looking at stars right before they die - predicting whether or not it will become a pulsar or not (could be slightly impossible considering stars live for billions  of years…)
