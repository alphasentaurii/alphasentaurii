---
layout: post
title:  "STARSKØPE: Cyberoptic Artificial Telescope"
date:   2020-04-04 04:04:04 -1111
categories: datascience
tags: astrophysics project machine-learning AI neural-networks starskope astronomy
author: Ru Keïn
---

> "Mathematicians [...] are often led astray when 'studying' physics because they lose sight of the physics. They say: *'Look, these differential equations--the Maxwell equations--are all there is to electrodynamics; it is admitted by the physicists that there is nothing which is not contained in the equations. The equations are complicated, but after all they are only mathematical equations and if I understand them mathematically inside out, I will understand the physics inside out.'* Only it doesn't work that way. Mathematicians who study physics with that point of view--and there have been many of them--usually make little contribution to physics and, in fact, little to mathematics. They fail because the actual physical situations in the real world are so complicated that it is necessary to have a much broader understanding of the equations."
**-Richard Feynman, *The Feynman Lectures on Physics: Volume 2*, Chapter 2-1: "Differential Calculus of Vector Fields"**

<div >
<img src="/assets/images/starskope/feynman-bongos.jpg" alt="Feynman Bongos" title="Richard Feynman" width="400"/>
</div>

> Feynman circa 1962 (photographer unknown) courtesy of Ralph Leighton. Source: _Feynman's Tips on Physics_ 

## So...do we need physics?

The problem with the dataset I'm using for this project is that it came **without any units**. How can we be expected to incorporate the laws of physics into any model if we don't know the units?? I wondered if it was some sort of test - that NASA put this dataset on Kaggle without any units because they wanted to see if a data scientist with some time on their hands (no one), or with an incurable sense of curiosity (most of us) would take it upon themselves (probably just me) to figure out the units themselves by doing some research and matching up the data points with the original Campaign 3 data on the Mikulsky Archive website, then using the AstroPy and AstroQuery libraries to perform phase-folding to identify transiting planets. Nevermind your project is due in one week and you won't graduate if it's late.

## That's not very nice.

Then I thought that's a bizarre task to put someone up to - not very nice of them to take away our precious units. How the heck do we do phase-folding, and how do we filter out junk data points where K2's thrusters were firing?? After going very deep into Fourier transforms to identify the period through harmonic means (because I'm a musician and that seemed like an excellent rabbit-hole to go down), I thought, well, maybe the real test is to see if we can build a model that filters out all of the non-candidates without having to go to all that trouble of period estimation and phase-folding. 

## ...But it worked anyway...maybe.

Unfortunately (perhaps) I was able to identify all 5 planets in the test set **without any physics**. I'm still not totally convinced, but I ultimately concluded that even though the model worked, I would need to validate its accuracy by testing against a larger data set and seeing if my model still performed well. I could then compare it to a model that uses the physics, and the unique properties of the telescope, to clean and scrub the data as well as perform normalization *properly*. This is why there is a STARSKØPE 2 and 3 which you can read about in the posts following this one.

<div style="max-width:100%">
<iframe src="https://player.vimeo.com/video/401277721" width="640" height="360" frameborder="0" allow="autoplay; fullscreen" allowfullscreen></iframe></div>

## Questions

Some specific questions this project seeks to answer are as follows: 

1. Can a transiting exoplanet be detected strictly by analyzing the raw flux values of a given star? 
    
2. What is the best approach for pre-processing photometric timeseries data and what are some of the issues we might encounter in choosing how the data is prepared for classification modeling?
    
3. How much signal-to-noise ratio is too much? That is, if the classes are highly imbalanced, for instance only a few planets can be confirmed out of thousands of stars, does the imbalance make for an unreliable or inaccurate model? 

4. How do we test and validate that?
  
<img src="/assets/images/starskope/288_planetbleed1600.jpeg" alt="planet threshold crossing event" title="Planet Threshold Crossing Event" width="400"/>
copyright: NASA

## W∆

The robot who posted the dataset on Kaggle (I assume she's a robot because her name is `Winter Delta` or `W∆`) gives us a few hints on how we *could* determine the units, and the dimensions, and a lot of other important physics-related information, if we do a little research. The biggest hint is that this dataset is from the K2 space telescope's Campaign 3 observations in which only 42 confirmed exoplanets are detected in a set of over 5,000 stars. Looking at the dataset on its own (before doing any digging), we are given little information about how long the time period covers, and we know do not know what the time intervals between flux values are. So far, this has not stopped any data scientists from attempting to tackle the classification model without gathering any additional information. 

# Signal Analysis - Star Flux Plots

The flux values for planet vs. no planet can in some cases be fairly obvious, although this is not always the case. One of the problems I encountered with having no units was not being able to perform proper de-noising techniques unique to the telescopic instrument (correct for smearing, thrusters firing, stellar variability). Additionally, this dataset does not allow us to do any astrophysical assessment of the transit event, for instance determining the period, predicting if it's an eclipsing binary vs. exoplanet etc. This particularity will be explored later using MAST data in the next phase of the project (see Starskope 3).

# Scatterplot of Star Flux Signals - Confirmed Exoplanet

<div>
<img src="/assets/images/starskope/output_32_0" alt="signal timeseries plot - planet" title="" width="400"/>
</div>

You can see above (and below - same signal using line plot) the values drop for a brief period of time then return. This could be a possible planet (in this case it is in fact a planet). 

# Lineplot of Star Flux Signals - Confirmed Exoplanet

<div>
<img src="/assets/images/starskope/output_32_1" alt="signal timeseries plot - planet" title="" width="400"/>
</div>

By contrast, the raw values of non-planet star flux values do reveal some important differences. Notice where on the y-axis the majority of values lay, and there is no significant drop indicating a transit. Some of these show a spike upwards, which could be explained by a number of things, potentially some kind of noise or disturbance interfering with the sensors either internally or from the sun.

# Scatterplot of Star Flux Signals with No Exoplanet

<div>
<img src="/assets/images/starskope/output_34_0" alt="signal timeseries plot" title="" width="400"/>
</div>

# No exoplanet - lineplot

<div>
<img src="/assets/images/starskope/output_34_1" alt="signal timeseries plot" title="" width="400"/>
</div>

Compare another two sets of flux values (blue has a transiting planet, red does not):

<div>
<img src="/assets/images/starskope/output_32_3" alt="signal timeseries plot" title="" width="400"/>
</div>

<div>
<img src="/assets/images/starskope/output_34_3" alt="signal timeseries plot" title="" width="400"/>
</div>

## Model

The baseline model I started with is a convolutional neural network using the `Keras API` in a `sci-kit learn`-esque wrapper. All the functions used in this project (as well as the extended Starskøpe project) are included in a python (PyPi) package I wrote called `spacekit`, and you can check out my repo [spacekit](https://github.com/alphasentaurii/spacekit) to view the source code. 

<div>
<img src="/assets/images/starskope/output_96_3" alt="Training Accuracy and Loss" title="Training Accuracy and Loss" width="400"/>
</div>

## Results

I was able to identify with 99% accuracy the handful of stars (5) in the test dataset that have a confirmed exoplanet in their orbit. This baseline model is mathematically accurate, but it does not "understand physics". The conclusion we need to make about the model is whether or not this lack of physics embedded in the training process (or even pre-training process) is acceptable or not.

## Fusion Matrix

<div>
<img src="/assets/images/starskope/output_96_0" alt="Fusion Matrix" title="Fusion Matrix" width="400"/>
</div>

## ROC Area Under Curve (AUC)

A little too perfect?

<div>
<img src="/assets/images/starskope/output_96_1" alt="ROC AUC" title="ROC AUC" width="400"/>
</div>

## Conclusion

1. While it is possible to create a 99% accurate machine learning model for detecting exoplanets using the raw flux values, without any sense of the actual time intervals, and with a highly imbalanced data set (meaning only a few positive examples in a sea of negatives) - it is unclear that we can "get away with" this in every case. 

2. Furthermore, it is unlikely that could feel completely sure that we aren't missing out on critical information - such as detecting the existence of an earth-like exoplanet transiting a star - if we don't use our understanding of physics to further de-noise, normalize, and scale the data before training the model (and possibly even embed this into a pre-training phase). 

3. As a case in point, if you read any of the space telescope handbooks, you will quickly learn just how complex the instruments that are producing this data are, and that the way their technology works, when and where in the sky they were pointing, as well as what actually happened during their missions, you'd know that should all probably be taken into account in your model! The K2 data in particular, for instance, has a unique issue that every so often its thrusters would fire to adjust/maintain its position in the sky, causing data at multiple points to be completely useless. 

## Why that matters...

This type of noise cannot be removed without knowing what exact times the thrusters fired, as well as what times each of the observations of the dataset occurred. Even if we do manage to throw the bad data out, we are still stuck with the problem of not having any data for that time period, and once again might miss our potential planet's threshold crossing event! **If we know where and when those missing pieces occur, we could use that to collect our missing data from another telescope like TESS, which has overlapping targets of observation.** 

## A Dynamic Model

A model that can combine data from two different space telescopes, and be smart enough to know based on the telescope it came from how to handle the data, would make truly accurate predictions, and much more useful classifications. 

## What we can do about that...

This is the type of model I will set out to build in my future work. This is what we would call a cyberoptic artificial telescope - one that can aggregate large datasets from multiple missions and give us a more accurate, more detailed picture of the stars and planets than what we have available to us in the limited view of a single picture from a single telescope at a single point in time. This is the vision for *STARSKØPE* which will come out of this project.

## Recommendations

My recommendations are the following:

1. Use datasets from the MAST website (via API) to incorporate other calculations of the star's properties as features to be used for classification algorithms. Furthermore, attempt other types of transformations and normalizations on the data before running the model - for instance, apply a Fourier transform.

2. Combine data from multiple campaigns and perhaps even multiple telescopes (for instance, matching sky coordinates and time intervals between K2, Kepler, and TESS for a batch of stars that have overlapping observations - this would be critical for finding transit periods that are longer than the campaigns of a single telecope's observation period).

3. Explore using computer vision on not only the Full Frame images we can collect from telescopes like TESS, but also on spectographs of the flux values themselves. The beauty of machine learning is our ability to rely on the computer to pick up very small nuances in differences that we ourselves cannot see with our own eyes. 

<div>
<img src="/assets/images/starskope/spec-transform.png" alt="fourier-transform spectrograph" title="Fourier-transform Spectrograph" width="400"/>
</div>

__Spectrograph of Fourier-transform Flux Signal__
   
4. Explore using autoencoded machine learning algorithms with Restricted Boltzmann Machines - this type of model has proven to be incredibly effective in the image analysis of handwriting as we've seen applied the MNIST dataset - let's find out if the same is true for images of stars, be they the Full Frame Images or spectographs.

## Future Work

To continue this project, I'll take another approach for detecting exoplanets using computer vision to analyze images of spectographs of this same star flux data set. Please go to the next post [starskope-2](/datascience/2020/05/06/starskope-2-spectrograph-image-classification.html) to see how I build a `convolutional neural network` to classify stars using spectrograph images of the flux values to find transiting exoplanets. Following this, I apply the same algorithm to spectrographs of Fourier transformed data.

In [starskope-3](/datascience/2020/06/01/starskope-3-scraping-mast-api.html) I scrape the MAST API data hosted on AWS via S3 to perform a more in-depth analysis using a physics-based machine learning model.

Additional future work following this project will be to develop my "cyberoptic artificial telescope" as a machine learning driven application that any astrophysicist can use to look at a single or collection of stars and have the model classify them according not only to exoplanet predictions, but also predict what type of star it is, and other key properties that would be of interest for astrophysical science applications.