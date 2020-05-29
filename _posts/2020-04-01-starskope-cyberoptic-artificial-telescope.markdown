---
layout: post
title:  "Starskøpe 1 - Cyberoptic Artificial Telescope"
date:   2020-04-01 11:11:11 -1111
categories: datascience
---

## Building a Cyberoptic Artificial Telescope for Astrophysical Object Classification
> "Mathematicians [...] are often led astray when 'studying' physics because they lose sight of the physics. They say: *'Look, these differential equations--the Maxwell equations--are all there is to electrodynamics; it is admitted by the physicists that there is nothing which is not contained in the equations. The equations are complicated, but after all they are only mathematical equations and if I understand them mathematically inside out, I will understand the physics inside out.'* Only it doesn't work that way. Mathematicians who study physics with that point of view--and there have been many of them--usually make little contribution to physics and, in fact, little to mathematics. They fail because the actual physical situations in the real world are so complicated that it is necessary to have a much broader understanding of the equations."
**-Richard Feynman, *The Feynman Lectures on Physics: Volume 2*, Chapter 2-1: "Differential Calculus of Vector Fields"**
>

<img src="/assets/images/starskope/feynman-bongos.jpg" width=400>

## INTRODUCTION

One of the reasons I quote Mr. Feynman above is because I set out to work on this project with only one year of high school physics under my belt. Despite loving the subject and even getting an A- in that one class, for some reason I did not continue pursuing physics while in school. I bought the Feynman lectures a few years back (on a whim? who does that?) and as soon as I began preparing for this project I felt intuitively that it would be somewhat ridiculous for me to build neural networks for classifying astrophysical data if I didn't fully grasp how and why the equations used to calculate my findings actually work.  



## QUESTIONS
The specific questions this project seeks to answer are as follows: 

1. Can a transiting exoplanet be detected strictly by analyzing the raw flux values of a given star? 
    
2. What is the best approach for pre-processing photometric timeseries data and what are some of the issues we might encounter in choosing how the data is prepared for classification modeling?
    
3. How much signal-to-noise ratio is too much? That is, if the classes are highly imbalanced, for instance only a few planets can be confirmed out of thousands of stars, does the imbalance make for an unreliable or inaccurate model? 

4. How do we test and validate that?
  

## DATASET

To answer the above questions, I started the analysis with a small labeled timeseries dataset from Kaggle posted by NASA several years ago. The reason I chose this particular dataset is because in terms of the type of information we typically need to know in order to solve a physics problem -- the primary one being UNITS, otherwise it's a math problem! -- this one is barren. The author who posted the dataset (`Winter Delta` or `W∆`) does however give us a few hints on how we *could* determine the units, and the dimensions, and a lot of other important physics-related information, if we do a little research. The biggest hint is that this dataset is from the K2 space telescope's Campaign 3 observations in which only 42 confirmed exoplanets are detected in a set of over 5,000 stars. Looking at the dataset on its own (before doing any digging), we are given little information about how long the time period covers, and we know do not know what the time intervals between flux values are. So far, this has not stopped any data scientists from attempting to tackle the classification model without gathering any additional information. 


<img src="/assets/images/starskope/288_planetbleed1600.jpeg" width=600>
source: NASA


## MODEL

To answer the question, I first set out to build a model for the data as is, "sans-physics". The baseline model is a neural network using the `Keras API` in a `sci-kit learn` wrapper.  

## RESULTS

I was able to identify with 99% accuracy the handful of stars (5) in the test dataset that have a confirmed exoplanet in their orbit. This baseline model is mathematically accurate, but it does not "understand physics". The conclusion we need to make about the model is whether or not this lack of physics embedded in the training process (or even pre-training process) is acceptable or not.

## CONCLUSION

While it is possible to create a 99% accurate machine learning model for detecting exoplanets using the raw flux values, without any sense of the actual time intervals, and with a highly imbalanced data set (imbalanced meaning only a few positive examples in a sea of negatives) - it is unclear that we can "get away with" this in every case. Furthermore, it is unlikely that could feel completely sure that we aren't missing out on critical information - such as detecting the existence of an earth-like exoplanet transiting a star - if we don't use our understanding of physics to further de-noise, normalize, and scale the data before training the model (and possibly even embed this into a pre-training phase). As a case in point, if you read any of the space telescope handbooks, you will quickly learn just how complex the instruments that are producng this data are, and that the way their technology works, when and where in the sky they were pointing, as well as what actually happened during their missions, you'd know that should all probably be taken into account in your model! The K2 data in particular, for instance, has a unique issue that every so often its thrusters would fire to adjust/maintain its position in the sky, causing data at multiple points to be completely useless. 

## Why that matters...

This type of noise cannot be removed without knowing what exact times the thrusters fired, as well as what times each of the observations of the dataset occurred. Even if we do manage to throw the bad data out, we are still stuck with the problem of not having any data for that time period, and once again might miss our potential planet's threshold crossing event! If we know where and when those missing pieces occur, we could use that to collect our missing data from another telescope like TESS, which has overlapping targets of observation. A model that can combine data from two different space telescopes, and be smart enough to know based on the telescope it came from how to handle the data, would make truly accurate predictions, and much more useful classifications. 

## What we can do about that...

This is the type of model I will set out to build in my future work. This is what we would call a cyberoptic artificial telescope - one that can aggregate large datasets from multiple missions and give us a more accurate, more detailed picture of the stars and planets than what we have available to us in the limited view of a single picture from a single telescope at a single point in time. This is the vision for *STARSKØPE* which will come out of this project.

## RECOMMENDATIONS

My recommendations are the following:

1. Use datasets from the MAST website (via API) to incorporate other calculations of the star's properties as features to be used for classification algorithms. Furthermore, attempt other types of transformations and normalizations on the data before running the model - for instance, apply a Fourier transform.

2. Combine data from multiple campaigns and perhaps even multiple telescopes (for instance, matching sky coordinates and time intervals between K2, Kepler, and TESS for a batch of stars that have overlapping observations - this would be critical for finding transit periods that are longer than the campaigns of a single telecope's observation period).

3. Explore using computer vision on not only the Full Frame images we can collect from telescopes like TESS, but also on spectographs of the flux values themselves. The beauty of machine learning is our ability to rely on the computer to pick up very small nuances in differences that we ourselves cannot see with our own eyes. 
   
4. Explore using autoencoded machine learning algorithms with Restricted Boltzmann Machines - this type of model has proven to be incredibly effective in the image analysis of handwriting as we've seen applied the MNIST dataset - let's find out if the same is true for images of stars, be they the Full Frame Images or spectographs.

## FUTURE WORK
To continue this project, I'll take another approach for detecting exoplanets using computer vision to analyze images of spectographs of this same star flux data set. Please go to the next post `[starskope-2](/datascience/2020/05/06/starskope-2-spectrograph-image-classification.html)` to see how I build a `convolutional neural network` to classify stars using spectrograph images of the flux values to find transiting exoplanets. Following this, I apply the same algorithm to spectrographs of Fourier transformed data.

Additional future work following this project will be to develop my "cyberoptic artificial telescope" as a machine learning driven application that any astrophysicist can use to look at a single or collection of stars and have the model classify them according not only to exoplanet predictions, but also predict what type of star it is, and other key properties that would be of interest for astrophysical science applications.

## SCRUB
**Initial inspection of data, reviewing the features, target (if any), datatypes, and checking for nulls.**

Each star's light frequency makes up a single row of data collected over the course of the campaign (#3), which in this case for K2 campaign 3 was a little over 60 days (campaigns are normally ~80 days but c3 ended early due to data storage capacity issues. 

If we crunched the numbers (which I did elsewhere), it's 29.4 minutes between each flux measurement, also known as the cadence. This matches the information available in the K2 handbook/MAST website/NASA. Knowing the units and time intervals would allow us to scale and normalize the data very methodically. However, since our initial (math-based) model doesn't 'care' about units, the scrubbing will not take any of the physics into account. This is intentional.

This is something we DO want to come back to for comparison with future models that *will* have the astrophysical properties embedded in their pre-learning process, and in particular the SCRUBBING: remember, this is a *timeseries*...it's hard to do any normalizing, scaling, de-noising to a timeseries if we don't know anything about the time units. And that's only ONE of the dimensions being completely ignored by our strict mathematical approach. The question is, will it matter? 

### Initial Inspection


```python
# Check the value counts 
display(train['LABEL'].value_counts(),test['LABEL'].value_counts())
```


    1    5050
    2      37
    Name: LABEL, dtype: int64



    1    565
    2      5
    Name: LABEL, dtype: int64

### TARGET
Our target column `LABEL` assigns each star with a 1 or a 2 to designate whether or not there is a confirmed exoplanet that was found in the star's orbit. This is precisely what we are trying to classify in our model below.

Notice there are a total of only 42 stars that are labeled "2", ie confirmed exoplanet orbiting this star. There are 37 exoplanet host stars in the training set, and only 5 in the test set. Such highly imbalanced classes will be something we need to deal with carefully in our model.

## Explore

### Planet Host vs Non-Host Stars

Since we are setting out to classify stars as being either a planet-host or non-host, it would be useful to compare the data visually and see if we can pick up on any significant differences in the flux values with just our eyeballs. The simplest way to do this is plot the signals of each target class for a couple of stars and look at the scatter plots and a line plots.

### Threshold Crossing Event (TCE)
TCE is determined by a significant dip in the flux values, the assumption being something crossed in front of the star blocking its light for some period of time that the telescope has designated as suspect of an orbiting planet! The occurrence of a TCE means that star is flagged as a 'Target of Interest' or in K2's case, 'Kepler Object of Ineterst' (KOI). The KOIs for each campaign have to be confirmed by a human, of course, usually an astrophysicist, and that is precisely where machine learning comes in - there are billions and billions of stars, and thus billions of billions of potential data points. "Looking for a needle in a haystack" doesn't even work as a metaphor for a scale this immense. This is the ultimate challenge for data scientists! Let's see what this looks like.


```python
# grab first row of observations to create pandas series 

# first row is label = 2 which is a confirmed exoplanet host star
# TCE "Threshold Crossing Event"
tce1 = train.iloc[0, :]
tce2 = train.iloc[1, :]

# last row is label = 1 (no tce meaning no evidence this star hosts a planet)
no_tce1 = train.iloc[-1, :]
no_tce2 = train.iloc[-2, :]

display(tce1.head(),tce2.head(),no_tce1.head(), no_tce2.head())
```


    LABEL      2.00
    FLUX.1    93.85
    FLUX.2    83.81
    FLUX.3    20.10
    FLUX.4   -26.98
    Name: 0, dtype: float64



    LABEL      2.00
    FLUX.1   -38.88
    FLUX.2   -33.83
    FLUX.3   -58.54
    FLUX.4   -40.09
    Name: 1, dtype: float64



    LABEL       1.00
    FLUX.1    323.28
    FLUX.2    306.36
    FLUX.3    293.16
    FLUX.4    287.67
    Name: 5086, dtype: float64



    LABEL     1.00
    FLUX.1    3.82
    FLUX.2    2.09
    FLUX.3   -3.29
    FLUX.4   -2.88
    Name: 5085, dtype: float64


### A Word on Units..

After doing a little research (mostly by reading the K2 Handbook and visiting the MAST website where NASA houses all of its space telescope data) we learn that the flux values for campaign 3 that are in the Kaggle dataset have been put through a de-noising process. Prior to this particular de-noising process, the flux values would be called `SAP Flux` however in this case we are dealing with `PDC_SAP Flux`. At the moment the units may not seem to matter much, since we assume they are consist across all observations. However, as with anything relating to physics, and science for that matter, the units MATTER. All that to say, for now we are at least going to label the axes accurately so that later down the line if we want to compare this dataset to another from the archive, we will know the units! :)


### atomic_vector_plotter()

```python

def atomic_vector_plotter(signal, label_col=None, classes=None, class_names=None, figsize=(15,5), 
     y_units=None, x_units=None):
         """
         Plots scatter and line plots of time series signal values.  
        
         **ARGS
         signal: pandas series or numpy array
         label_col: name of the label column if using labeled pandas series
             -use default None for numpy array or unlabeled series.
             -this is simply for customizing plot Title to include classification    
         classes: (optional- req labeled data) tuple if binary, array if multiclass
         class_names: tuple or array of strings denoting what the classes mean
         figsize: size of the figures (default = (15,5))
        
         ******
        
        Ex1: Labeled timeseries passing 1st row of pandas dataframe
         > first create the signal:
         star_signal_alpha = x_train.iloc[0, :]
         > then plot:
         star_signals(star_signal_alpha, label_col='LABEL',classes=[1,2], 
                     class_names=['No Planet', 'Planet']), figsize=(15,5))
       
         Ex2: numpy array without any labels
         > first create the signal:
        
         >then plot:
         star_signals(signal, figsize=(15,5))
        
         ######
         TODO: 
         -`signal` should take an array rather than pdseries
         -could allow either array or series to be passed, conv to array if series 
         ######
         """
        
         # pass None to label_col if unlabeled data, creates generic title
         if label_col is None:
             label = None
             title_scatter = "Scatterplot of Star Flux Signals"
             title_line = "Line Plot of Star Flux Signals"
             color='black'
            
         # store target column as variable 
         elif label_col is not None:
             label = signal[label_col]
             # for labeled timeseries
             if label == 1:
                 cn = class_names[0]
                 color='red'

             elif label == 2:
                 cn = class_names[1] 
                color='blue'
         #create appropriate title acc to class_names    
             title_scatter = f"Scatterplot for Star Flux Signal: {cn}"
             title_line = f"Line Plot for Star Flux Signal: {cn}"
        
         # Set x and y axis labels according to units
         # if the units are unknown, we will default to "Flux"
         if y_units == None:
             y_units = 'Flux'
         else:
             y_units = y_units
         # it is assumed this is a timeseries, default to "time"   
         if x_units == None:
             x_units = 'Time'
         else:
             x_units = x_units
        
         # Scatter Plot 
        
         plt.figure(figsize=figsize)
         plt.scatter(pd.Series([i for i in range(1, len(signal))]), 
                     signal[1:], marker=4, color=color)
         plt.ylabel(y_units)
         plt.xlabel(x_units)
         plt.title(title_scatter)
         plt.show()

         # Line Plot
         plt.figure(figsize=figsize)
         plt.plot(pd.Series([i for i in range(1, len(signal))]), 
                 signal[1:], color=color)
         plt.ylabel(y_units)
         plt.xlabel(x_units)
         plt.title(title_line)
         plt.show()
```


```python
from spacekit.analyzer import Flux

Flux.atomic_vector_plotter(signal=tce1, label_col='LABEL', classes=[1,2], 
         class_names=['No Planet', 'Planet'], 
         y_units='PDC_SAP Flux', x_units='Time')

Flux.atomic_vector_plotter(signal=tce2, label_col='LABEL', classes=[1,2],
         class_names=['No Planet', 'Planet'],
         y_units='PDC_SAP Flux', x_units='Time') 
```

<img src="/assets/images/starskope/output_32_0.png" width=400>


<img src="/assets/images/starskope/output_32_1.png" width=400>


<img src="/assets/images/starskope/output_32_2.png" width=400>


<img src="/assets/images/starskope/output_32_3.png" width=400>

This second star's flux signal pattern looks very different - are we to assume that each one of those dips is a transit event? Perhaps more than one planet is orbiting? Otherwise that would be a fairly short period. Let's compare these to the NON planet host stars:

```python
Flux.atomic_vector_plotter(signal=no_tce1, label_col='LABEL', classes=[1,2],
                         class_names=['No Planet', 'Planet'],
                          y_units='PDC_SAP Flux', x_units='Time')

Flux.atomic_vector_plotter(signal=no_tce2, label_col='LABEL', classes=[1,2],
                         class_names=['No Planet', 'Planet'],
                          y_units='PDC_SAP Flux', x_units='Time') 
```

<img src="/assets/images/starskope/output_34_0.png" width=400>

<img src="/assets/images/starskope/output_34_1.png" width=400>

<img src="/assets/images/starskope/output_34_2.png" width=400>

<img src="/assets/images/starskope/output_34_3.png" width=400>

It's hard to make a fair comparison with these plots without being able to see much in detail. We need to "zoom in" - this can be accomplished through normalizing and scaling techniques, but the standard procedure for this type of data would be to perform `phase-folding` based on the estimated period of the transiting planets.

## Pre-processing


```python
from spacekit.transformer import Transformer
T = transformer.Transformer()
```

### hypersonic_pliers()

```python


def hypersonic_pliers(path_to_train, path_to_test):
        
         """
         Using Numpy to extract data into 1-dimensional arrays
         separate target classes (y) for training and test data
         assumes y (target) is first column in dataframe
       
         #TODO: option to pass in column index loc for `y` if not default (0) 
         #TODO: option for `y` to already be 0 or 1 (don't subtract 1)
         #TODO: allow option for `y` to be categorical (convert via binarizer)
         """
         import numpy as np
       
         Train = np.loadtxt(path_to_train, skiprows=1, delimiter=',')
         X_train = Train[:, 1:]
         y_train = Train[:, 0, np.newaxis] - 1.
       
         Test = np.loadtxt(path_to_test, skiprows=1, delimiter=',')
         X_test = Test[:, 1:]
         y_test = Test[:, 0, np.newaxis] - 1.
        
         del Train,Test
         print("X_train: ", X_train.shape)
         print("y_train: ", y_train.shape)
         print("X_test: ", X_test.shape)
         print("y_test: ", y_test.shape)
        
         return X_train, X_test, y_train, y_test
```


```python
X_train,X_test,y_train,y_test = T.hypersonic_pliers(DATA+'/exoTrain.csv', 
                                          DATA+'/exoTest.csv') 
```

    X_train:  (5087, 3197)
    y_train:  (5087, 1)
    X_test:  (570, 3197)
    y_test:  (570, 1)


### Scaling: thermo_fusion_chisel()

Scale each observation to zero mean and unit variance.



```python


def thermo_fusion_chisel(matrix1, matrix2=None):
             """
             Scales each array of a matrix to zero mean and unit variance.
             returns matrix/matrices of same shape as input but scaled
             matrix2 is optional - useful if data was already train-test split
             example: matrix1=X_train, matrix2=X_test
           
             """
             import numpy as np
           
               
             matrix1 = ((matrix1 - np.mean(matrix1, axis=1).reshape(-1,1)) / 
                 np.std(matrix1, axis=1).reshape(-1,1))
           
             print("Mean: ",matrix1[0].mean())
             print("Variance: ",matrix1[0].std())
           
             if matrix2 is not None:
                 matrix2 = ((matrix2 - np.mean(matrix2, axis=1).reshape(-1,1)) / 
                     np.std(matrix2, axis=1).reshape(-1,1))
           
                 print("Mean: ",matrix2[0].mean())
                 print("Variance: ",matrix2[0].std())
                 return matrix1,matrix2
             else:
                 return matrix1
```


```python
# Scale each row to zero mean and unit variance.
X_train, X_test = T.thermo_fusion_chisel(X_train, X_test)
```

    Mean:  0.0
    Variance:  1.0
    Mean:  2.6670356049800446e-17
    Variance:  1.0


### De-noising: babel_fish_dispenser()

In order to reduce the amount of high frequency noise that is likely to have an adverse effect on the neural network's learning outcomes, we can pass a uniform 1-D filter on our scaled train and test data then stack the arrays along the second axis. There are other techniques we might want to apply for further de-noising but for now we'll start with this for the baseline.


```python
print(X_train.shape)
print(y_train.shape)
```

    (5087, 3197)
    (5087, 1)



```python


def babel_fish_dispenser(matrix1, matrix2=None, step_size=None, axis=2):
         """        
         Adds an input corresponding to the running average over a set number
         of time steps. This helps the neural network to ignore high frequency 
         noise by passing in a uniform 1-D filter and stacking the arrays. 
         **ARGS
         step_size: integer, # timesteps for 1D filter. defaults to 200
         axis: which axis to stack the arrays

         ex:
         noise_filter(matrix1=X_train, matrix2=X_test, step_size=200)
         """
         import numpy as np
         from scipy.ndimage.filters import uniform_filter1d

         if step_size is None:
             step_size=200

         # calc input for flux signal rolling avgs 
         filter1 = uniform_filter1d(matrix1, axis=1, size=step_size)
         # store in array and stack on 2nd axis for each obs of X data
         matrix1 = np.stack([matrix1, filter1], axis=2)

         if matrix2 is not None:
             filter2 = uniform_filter1d(matrix2, axis=1, size=step_size)
             matrix2 = np.stack([matrix2, filter2], axis=2)
             print(matrix1.shape,matrix2.shape)
             return matrix1,matrix2
         else:
             print(matrix1.shape)
             return matrix1
```


```python
# we now have a 2-dimensional array for every star
X_train, X_test = T.babel_fish_dispenser(X_train, X_test, step_size=200, 
                                         axis=2)
```

    (5087, 3197, 2) (570, 3197, 2)



```python
# array on 2nd axis
print('\nx_train[-1] flux signal rolling avgs\n')
# plot arrays
rolling = X_train[1][:,1]
print(rolling)
plt.plot(rolling)
```

    
    x_train[-1] flux signal rolling avgs
    
    [-0.10910981 -0.10801068 -0.10926314 ... -0.18190533 -0.19232921
     -0.21176035]
<img src="/assets/images/starskope/output_48_2.png" width=400>

```python
# viewed together...
plt.plot(X_train[1][:,0])
plt.plot(rolling)
```

<img src="/assets/images/starskope/output_49_1.png" width=400>

## Model

### **Tactical Decisions**

Since I'm building the baseline model from scratch, a few considerations need to be made. While we can run a gridsearch (or randomizedsearchCV) to get the parameters for us, we still need to decide what type of model would be most ideal for this dataset, knowing what we know so far based on the work done so far. From there, we can go with best practices, assess the initial outcomes, and tune the hyperparameters with each iteration. 

### **CNN**
The baseline will consist of a one-dimensional convolutional neural network (CNN). This is ideal for working with this particular dataset in which we will pass one row of the timeseries flux values as an array. This is very similar to how we would process image data (and that's strategically useful if we want to develop the model in the future to handle Full-Frame Images from Tess, for instance, or spectographs of the flux frequences, for instance. 

### **1-Layer at a time**
We'll be using the Keras API which makes it easy to add in the layers one at a time. Each 1D convolutional layer corresponds to a local filter, and then a pooling layer reduces the data length by approximately a factor 4. At the end, there are two dense layers. Again, this is similar to the approach taken for a typical image classifier. 

### **Activation Function**
The RELU activation function is closest to how real neurons actually work and often produces the best results compared to the other options, so we'll at least start with this for the baseline.

### **Batch Normalization**
Finally, the batch normalization layers are what help to speed up convergence. 

## `Model 1`

We'll begin creating a baseline model with a lower than usual learning rate and then speed things up and fine-tune parameters for optimization in the next iterations. (The lower learning rate will help to ensure convergence.) 

We'll increase the learning rate in Model2 iteration and also tune any other parameters as necessary. The first iteration uses the Adam optimizer, however, SGD is also a good option we could try here.


```python
X_train.shape
```




    (5087, 3197, 2)




```python
y_train.shape
```




    (5087, 1)



## build_cnn()


```python
from spacekit.builder import Keras
K = builder.Keras()
```


```python

def build_cnn(X_train, X_test, y_train, y_test, kernel_size=None, 
               activation=None, input_shape=None, strides=None, 
               optimizer=Adam, learning_rate=None, loss=None, metrics=None):
     """
     Builds, compiles and fits a linear CNN using Keras API
     """
     import keras
     from keras.utils.np_utils import to_categorical
     from keras import models, layers, optimizers
     from keras.models import Sequential, Model
     from keras.layers import Conv1D, MaxPool1D, Dense, Dropout, Flatten, \
     BatchNormalization, Input, concatenate, Activation
     from keras.optimizers import Adam
     from keras.wrappers.scikit_learn import KerasClassifier
     from sklearn.model_selection import cross_val_score

     if input_shape is None:
         input_shape = X_train.shape[1:]
     if kernel_size is None:
         kernel_size=11
     if activation is None:
         activation='relu'
     if strides is None:
         strides = 4
     if learning_rate is None:
         learning_rate = 1e-5
     if loss is None:
         loss='binary_crossentropy'
     if metrics is None:
         metrics=['accuracy']

     print("BUILDING MODEL...")
     model=Sequential()

     print("LAYER 1")
     model.add(Conv1D(filters=8, kernel_size=kernel_size, 
                      activation=activation, input_shape=input_shape))
     model.add(MaxPool1D(strides=strides))
     model.add(BatchNormalization())

     print("LAYER 2")
     model.add(Conv1D(filters=16, kernel_size=kernel_size, 
                     activation=activation))
     model.add(MaxPool1D(strides=strides))
     model.add(BatchNormalization())

     print("LAYER 3")
     model.add(Conv1D(filters=32, kernel_size=kernel_size, 
                     activation=activation))
     model.add(MaxPool1D(strides=strides))
     model.add(BatchNormalization())

     print("LAYER 4")
     model.add(Conv1D(filters=64, kernel_size=kernel_size, 
                     activation=activation))
     model.add(MaxPool1D(strides=strides))
     model.add(Flatten())

     print("FULL CONNECTION")
     model.add(Dropout(0.5))
     model.add(Dense(64, activation=activation))
     model.add(Dropout(0.25))
     model.add(Dense(64, activation=activation))

     print("ADDING COST FUNCTION")
     model.add(Dense(1, activation='sigmoid'))

     ##### COMPILE #####
     model.compile(optimizer=optimizer(learning_rate), loss=loss, 
                 metrics=metrics)
     print("COMPILED")  

     return model 
```


```python
m1 = K.build_cnn(X_train, X_test, y_train, y_test, kernel_size=11, 
                     activation='relu', input_shape=X_train.shape[1:], 
                     strides=4, optimizer=Adam, learning_rate=1e-5, 
                     loss='binary_crossentropy', metrics=['accuracy'])
```

    BUILDING MODEL...
    LAYER 1
    LAYER 2
    LAYER 3
    LAYER 4
    FULL CONNECTION
    ADDING COST FUNCTION
    COMPILED


## Batch Generator

To correct for the extremely unbalanced dataset, we'll ensure that the network sees 50% of the positive sample over each batch. We will also apply augmentation by rotating each of the samples randomly each time, thus generating new data. This is similar to image classification when we rotate or shift the samples each time.

### fit_cnn()

```python


def fit_cnn(X_train,y_train, X_test, y_test, model, validation_data=None, 
                 verbose=None, epochs=None, steps_per_epoch=None, batch_size=None):
       
         if verbose is None:
             verbose=2
         if epochs is None:
             epochs = 5
         if validation_data is None:
             validation_data=(X_test, y_test)
         if steps_per_epoch is None:
             steps_per_epoch = (X_train.shape[1]//batch_size)
         if batch_size is None:
             batch_size = 32

         print("FITTING MODEL...")
        
         def batch_maker(X_train, y_train, batch_size=batch_size):
                 """
                 Gives equal number of positive and negative samples rotating randomly                
                 The output of the generator must be either
                 - a tuple `(inputs, targets)`
                 - a tuple `(inputs, targets, sample_weights)`.

                 This tuple (a single output of the generator) makes a single
                 batch. Therefore, all arrays in this tuple must have the same
                 length (equal to the size of this batch). Different batches may have 
                 different sizes. 

                 For example, the last batch of the epoch is commonly smaller than the others, 
                 if the size of the dataset is not divisible by the batch size.
                 The generator is expected to loop over its data indefinitely. 
                 An epoch finishes when `steps_per_epoch` batches have been seen by the model.
                
                 """
                 import numpy
                 import random
                 # hb: half-batch
                 hb = batch_size // 2
                
                 # Returns a new array of given shape and type, without initializing.
                 # x_train.shape = (5087, 3197, 2)
                 xb = np.empty((batch_size, X_train.shape[1], X_train.shape[2]), dtype='float32')
               
                 #y_train.shape = (5087, 1)
                 yb = np.empty((batch_size, y_train.shape[1]), dtype='float32')
                
                 pos = np.where(y_train[:,0] == 1.)[0]
                 neg = np.where(y_train[:,0] == 0.)[0]

                 # rotating each of the samples randomly
                 while True:
                     np.random.shuffle(pos)
                     np.random.shuffle(neg)
                
                     xb[:hb] = X_train[pos[:hb]]
                     xb[hb:] = X_train[neg[hb:batch_size]]
                     yb[:hb] = y_train[pos[:hb]]
                     yb[hb:] = y_train[neg[hb:batch_size]]
                
                     for i in range(batch_size):
                         size = np.random.randint(xb.shape[1])
                         xb[i] = np.roll(xb[i], size, axis=0)
               
                     yield xb, yb
        
         history = model.fit_generator(batch_maker(X_train, y_train, batch_size),
                                         validation_data=validation_data, 
                                         verbose=verbose, epochs=epochs, 
                                         steps_per_epoch=steps_per_epoch)
         print("TRAINING COMPLETE")
         model.summary()

         return history
```


```python
h1 = K.fit_cnn(X_train,y_train, X_test, y_test, m1,
               validation_data=(X_test, y_test), verbose=2, epochs=5, 
               steps_per_epoch=(X_train.shape[1]//32), batch_size=32)
```

    FITTING MODEL...
    Epoch 1/5
     - 26s - loss: 0.7978 - accuracy: 0.5044 - val_loss: 0.6152 - val_accuracy: 0.9351
    Epoch 2/5
     - 24s - loss: 0.7549 - accuracy: 0.5117 - val_loss: 0.6112 - val_accuracy: 0.8439
    Epoch 3/5
     - 26s - loss: 0.7197 - accuracy: 0.5319 - val_loss: 0.6259 - val_accuracy: 0.7386
    Epoch 4/5
     - 26s - loss: 0.7358 - accuracy: 0.5063 - val_loss: 0.6466 - val_accuracy: 0.6474
    Epoch 5/5
     - 23s - loss: 0.7300 - accuracy: 0.5142 - val_loss: 0.6549 - val_accuracy: 0.6193
    TRAINING COMPLETE
    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 output Shape              Param #   
    =================================================================
    conv1d_1 (Conv1D)            (None, 3187, 8)           184       
    _________________________________________________________________
    max_pooling1d_1 (MaxPooling1 (None, 797, 8)            0         
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 797, 8)            32        
    _________________________________________________________________
    conv1d_2 (Conv1D)            (None, 787, 16)           1424      
    _________________________________________________________________
    max_pooling1d_2 (MaxPooling1 (None, 197, 16)           0         
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 197, 16)           64        
    _________________________________________________________________
    conv1d_3 (Conv1D)            (None, 187, 32)           5664      
    _________________________________________________________________
    max_pooling1d_3 (MaxPooling1 (None, 47, 32)            0         
    _________________________________________________________________
    batch_normalization_3 (Batch (None, 47, 32)            128       
    _________________________________________________________________
    conv1d_4 (Conv1D)            (None, 37, 64)            22592     
    _________________________________________________________________
    max_pooling1d_4 (MaxPooling1 (None, 9, 64)             0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 576)               0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 576)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 64)                36928     
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 64)                0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 64)                4160      
    _________________________________________________________________
    dense_3 (Dense)              (None, 1)                 65        
    =================================================================
    Total params: 71,241
    Trainable params: 71,129
    Non-trainable params: 112
    _________________________________________________________________


## Evaluate (M1)

Let's assess the model thus far before tuning parameters. We'll create a few helper functions for calculating metrics and analyzing results visually. 

## Class Predictions




```python
# View function: get_preds()

# def get_preds(X,y,model=None,verbose=False):
#     if model is None:
#         model=model
#     # class predictions 
#     y_true = y.flatten()
    
#     y_pred = model.predict_classes(X).flatten() 
#     preds = pd.Series(y_pred).value_counts(normalize=False)
    
#     if verbose:
#         print(f"y_pred:\n {preds}")
#         print("\n")

#     return y_true, y_pred
```


```python
y_true, y_pred = computer.get_preds(X_test,y_test, model=m1,verbose=True)
```

    y_pred:
     0    354
    1    216
    dtype: int64
    
    


### False -/+ Rates (Training): fnfp()


```python


def fnfp(X,y,model, training=False):

     import numpy as np

     y_pred = np.round( model.predict(X) )

     pos_idx = y==1
     neg_idx = y==0

     #tp = np.sum(y_pred[pos_idx]==1)/y_pred.shape[0]
     fn = np.sum(y_pred[pos_idx]==0)/y_pred.shape[0]

     #tn = np.sum(y_pred[neg_idx]==0)/y_pred.shape[0]
     fp = np.sum(y_pred[neg_idx]==1)/y_pred.shape[0]

     if training:
         print(f"FN Rate (Training): {round(fn*100,4)}%")
         print(f"FP Rate (Training): {round(fp*100,4)}%")
     else:
         print(f"FN Rate (Test): {round(fn*100,4)}%")
         print(f"FP Rate (Test): {round(fp*100,4)}%")
```


```python
computer.fnfp(X_train, y_train, m1, training=True)
```

    FN Rate (Training): 0.2556%
    FP Rate (Training): 39.4339%


### False -/+ Rates (Test)


```python
computer.fnfp(X_test, y_test, m1)
```

    FN Rate (Test): 0.5263%
    FP Rate (Test): 37.5439%


### Classification Report

Sci-kit learn has a nice built-in method for evaluating our model:


```python
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, recall_score

report = metrics.classification_report(y_test,y_pred)
print(report)
```

                  precision    recall  f1-score   support
    
             0.0       0.99      0.62      0.76       565
             1.0       0.01      0.40      0.02         5
    
        accuracy                           0.62       570
       macro avg       0.50      0.51      0.39       570
    weighted avg       0.98      0.62      0.76       570
    


#### Fowlkes-Mallows

Fowlkes-Mallows is a good metric for imbalanced datasets, along with Jaccard which is similar to F1.


```python
sklearn.metrics.fowlkes_mallows_score(y_true,y_pred)
```




    0.7207089012303081



### Interpret Scores
With only 5 epochs, the model performed high in precision. However, because this such an imbalanced dataset, recall and F1 score are more critical metrics and these could definitely be improved. We'll tune some of the hyperparameters, specifically adjusting the learning rate and increasing the number of epochs up to 40. 

### History Metrics: keras_history()

The baseline model is not meant to give us optimal results - the real test will be in our final model below. First let's take a look at some of the visuals to understand what the scores really mean. This will help us decide how to proceed in tuning the model appropriately.


```python

def keras_history(history, figsize=(10,4)):
     """
     side by side sublots of training val accuracy and loss (left and right respectively)
     """
    
     import matplotlib.pyplot as plt
   
     fig,axes=plt.subplots(ncols=2,figsize=(15,6))
     axes = axes.flatten()
     ax = axes[0]
     ax.plot(history.history['accuracy'])
     ax.plot(history.history['val_accuracy'])
     ax.set_title('Model Accuracy')
     ax.set_ylabel('Accuracy')
     ax.set_xlabel('Epoch')
     ax.legend(['Train', 'Test'], loc='upper left')

     ax = axes[1]
     ax.plot(history.history['loss'])
     ax.plot(history.history['val_loss'])
     ax.set_title('Model Loss')
     ax.set_ylabel('Loss')
     ax.set_xlabel('Epoch')
     ax.legend(['Train', 'Test'], loc='upper left')
     plt.show()
```


```python
computer.keras_history(h1)
```


<img src="/assets/images/starskope/output_80_0.png" width=400>


With only a few epochs, and a small learning rate, it's obvious that our training parameters have a great deal of room for improvement. This is good - we will definitely need to adjust the learning rate. If that doesn't go far enough in producing desired results, we can also try using a different optimizer such as SGD instead of Adam. For now let's look at a few other key metrics.

## Fusion Matrix

It's like a confusion matrix, without the confusion...


```python

def fusion_matrix(matrix, classes=None, normalize=True, title='Fusion Matrix', cmap='Blues',
     print_raw=False): 
     """
     FUSION MATRIX!
     -------------
     It's like a confusion matrix...without the confusion.
   
     matrix: can pass in matrix or a tuple (ytrue,ypred) to create on the fly 
     classes: class names for target variables
     """
     from sklearn import metrics                       
     from sklearn.metrics import confusion_matrix #ugh
     import itertools
     import numpy as np
     import matplotlib as mpl
     import matplotlib.pyplot as plt
   
     # make matrix if tuple passed to matrix:
     if isinstance(matrix, tuple):
         y_true = matrix[0].copy()
         y_pred = matrix[1].copy()
        
         if y_true.ndim>1:
             y_true = y_true.argmax(axis=1)
         if y_pred.ndim>1:
             y_pred = y_pred.argmax(axis=1)
         fusion = metrics.confusion_matrix(y_true, y_pred)
     else:
         fusion = matrix
    
     # INTEGER LABELS
     if classes is None:
         classes=list(range(len(matrix)))

     #NORMALIZING
     # Check if normalize is set to True
     # If so, normalize the raw fusion matrix before visualizing
     if normalize:
         fusion = fusion.astype('float') / fusion.sum(axis=1)[:, np.newaxis]
         thresh = 0.5
         fmt='.2f'
     else:
         fmt='d'
         thresh = fusion.max() / 2.
    
     # PLOT
     fig, ax = plt.subplots(figsize=(10,10))
     plt.imshow(fusion, cmap=cmap, aspect='equal')
    
     # Add title and axis labels 
     plt.title(title) 
     plt.ylabel('TRUE') 
     plt.xlabel('PRED')
    
     # Add appropriate axis scales
     tick_marks = np.arange(len(classes))
     plt.xticks(tick_marks, classes, rotation=45)
     plt.yticks(tick_marks, classes)
    
     # Text formatting
     fmt = '.2f' if normalize else 'd'
     # Add labels to each cell
     #thresh = fusion.max() / 2.
     # iterate thru matrix and append labels  
     for i, j in itertools.product(range(fusion.shape[0]), range(fusion.shape[1])):
         plt.text(j, i, format(fusion[i, j], fmt),
                 horizontalalignment='center',
                 color='white' if fusion[i, j] > thresh else 'black',
                 size=14, weight='bold')
    
     # Add a legend
     plt.colorbar()
     plt.show() 
     return fusion, fig
```


```python
m1_fusion = computer.fusion_matrix(matrix=(y_true,y_pred), 
                          classes=['No Planet','Planet'], 
                                   title='M1 Fusion Matrix')
```


<img src="/assets/images/starskope/output_84_0.png" width=400>

The baseline model only managed to correctly identify 2 planets in the test set, while missing the other 3. The model incorrectly classified 215 non-TCEs as planets. 

## ROC AUC: roc_plots()

Plot the ROC area under the curve


```python


def roc_plots(X,y,model):
     from sklearn import metrics
     from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

     y_true = y.flatten()
     y_hat = model.predict(X)

     fpr, tpr, thresholds = roc_curve(y_true, y_hat) 

     # Threshold Cutoff for predictions
     crossover_index = np.min(np.where(1.-fpr <= tpr))
     crossover_cutoff = thresholds[crossover_index]
     crossover_specificity = 1.-fpr[crossover_index]

     fig,axes=plt.subplots(ncols=2, figsize=(15,6))
     axes = axes.flatten()

     ax=axes[0]
     ax.plot(thresholds, 1.-fpr)
     ax.plot(thresholds, tpr)
     ax.set_title("Crossover at {0:.2f}, Specificity {1:.2f}".format(crossover_cutoff, crossover_specificity))

     ax=axes[1]
     ax.plot(fpr, tpr)
     ax.set_title("ROC area under curve: {0:.2f}".format(roc_auc_score(y_true, y_hat)))
     plt.show()
    
     roc = roc_auc_score(y_true,y_hat)

     return roc
```


```python
m1_roc = computer.roc_plots(X_test, y_test, m1)
```

<img src="/assets/images/starskope/output_88_0.png" width=400>


# `Model 2`

Initial parameter tuning: increase learning rate to 3e-4 (0.0003), and increase epochs to 20. 

## Build M2


```python
#### MODEL 2 
# adjust learning rate to 3e-4

m2 = K.build_cnn(X_train, X_test, y_train, y_test, kernel_size=11, 
                     activation='relu', input_shape=X_train.shape[1:], 
                     strides=4, optimizer=Adam, learning_rate=3e-4, 
                     loss='binary_crossentropy', metrics=['accuracy'])
```

    BUILDING MODEL...
    LAYER 1
    LAYER 2
    LAYER 3
    LAYER 4
    FULL CONNECTION
    ADDING COST FUNCTION
    COMPILED


## Fit M2


```python
# increase number of epochs to 20

h2 = K.fit_cnn(X_train,y_train, X_test, y_test, m2,
               validation_data=(X_test, y_test), verbose=2, epochs=20, 
               steps_per_epoch=(X_train.shape[1]//32), batch_size=32)
```

    FITTING MODEL...
    Epoch 1/20
     - 22s - loss: 0.6945 - accuracy: 0.5827 - val_loss: 0.9014 - val_accuracy: 0.0193
    Epoch 2/20
     - 24s - loss: 0.6220 - accuracy: 0.6493 - val_loss: 0.8088 - val_accuracy: 0.2842
    Epoch 3/20
     - 26s - loss: 0.5791 - accuracy: 0.7033 - val_loss: 0.6019 - val_accuracy: 0.7175
    Epoch 4/20
     - 27s - loss: 0.5302 - accuracy: 0.7276 - val_loss: 0.4064 - val_accuracy: 0.8509
    Epoch 5/20
     - 29s - loss: 0.4857 - accuracy: 0.7715 - val_loss: 0.5536 - val_accuracy: 0.7140
    Epoch 6/20
     - 25s - loss: 0.4389 - accuracy: 0.7986 - val_loss: 0.4794 - val_accuracy: 0.7316
    Epoch 7/20
     - 24s - loss: 0.4059 - accuracy: 0.8220 - val_loss: 0.6372 - val_accuracy: 0.6351
    Epoch 8/20
     - 29s - loss: 0.3419 - accuracy: 0.8488 - val_loss: 0.7515 - val_accuracy: 0.5930
    Epoch 9/20
     - 28s - loss: 0.3195 - accuracy: 0.8665 - val_loss: 0.4455 - val_accuracy: 0.7667
    Epoch 10/20
     - 27s - loss: 0.2705 - accuracy: 0.8971 - val_loss: 0.8245 - val_accuracy: 0.6070
    Epoch 11/20
     - 28s - loss: 0.2380 - accuracy: 0.9066 - val_loss: 0.2590 - val_accuracy: 0.8789
    Epoch 12/20
     - 23s - loss: 0.2056 - accuracy: 0.9274 - val_loss: 0.3441 - val_accuracy: 0.8684
    Epoch 13/20
     - 41s - loss: 0.1805 - accuracy: 0.9280 - val_loss: 0.1826 - val_accuracy: 0.9298
    Epoch 14/20
     - 30s - loss: 0.1878 - accuracy: 0.9397 - val_loss: 0.1023 - val_accuracy: 0.9684
    Epoch 15/20
     - 23s - loss: 0.1755 - accuracy: 0.9451 - val_loss: 0.1844 - val_accuracy: 0.9404
    Epoch 16/20
     - 23s - loss: 0.1658 - accuracy: 0.9400 - val_loss: 0.3103 - val_accuracy: 0.8719
    Epoch 17/20
     - 23s - loss: 0.1483 - accuracy: 0.9501 - val_loss: 0.1477 - val_accuracy: 0.9526
    Epoch 18/20
     - 23s - loss: 0.1628 - accuracy: 0.9470 - val_loss: 0.1443 - val_accuracy: 0.9439
    Epoch 19/20
     - 29s - loss: 0.1118 - accuracy: 0.9631 - val_loss: 0.1330 - val_accuracy: 0.9614
    Epoch 20/20
     - 31s - loss: 0.1173 - accuracy: 0.9580 - val_loss: 0.0629 - val_accuracy: 0.9842
    TRAINING COMPLETE
    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 output Shape              Param #   
    =================================================================
    conv1d_5 (Conv1D)            (None, 3187, 8)           184       
    _________________________________________________________________
    max_pooling1d_5 (MaxPooling1 (None, 797, 8)            0         
    _________________________________________________________________
    batch_normalization_4 (Batch (None, 797, 8)            32        
    _________________________________________________________________
    conv1d_6 (Conv1D)            (None, 787, 16)           1424      
    _________________________________________________________________
    max_pooling1d_6 (MaxPooling1 (None, 197, 16)           0         
    _________________________________________________________________
    batch_normalization_5 (Batch (None, 197, 16)           64        
    _________________________________________________________________
    conv1d_7 (Conv1D)            (None, 187, 32)           5664      
    _________________________________________________________________
    max_pooling1d_7 (MaxPooling1 (None, 47, 32)            0         
    _________________________________________________________________
    batch_normalization_6 (Batch (None, 47, 32)            128       
    _________________________________________________________________
    conv1d_8 (Conv1D)            (None, 37, 64)            22592     
    _________________________________________________________________
    max_pooling1d_8 (MaxPooling1 (None, 9, 64)             0         
    _________________________________________________________________
    flatten_2 (Flatten)          (None, 576)               0         
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 576)               0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 64)                36928     
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 64)                0         
    _________________________________________________________________
    dense_5 (Dense)              (None, 64)                4160      
    _________________________________________________________________
    dense_6 (Dense)              (None, 1)                 65        
    =================================================================
    Total params: 71,241
    Trainable params: 71,129
    Non-trainable params: 112
    _________________________________________________________________


## Evaluate M2: compute()

The `compute` function combines all the functions used above for calculating the metrics into one shot:


```python

def compute(X, y, model, hist=None, preds=True, summary=True, fusion=True, 
             classes=None, report=True, roc=True):
     """
     evaluates model predictions and stores the output in a dict
     returns `results`
     """
     import pandas as pd
     import matplotlib.pyplot as plt
     from sklearn import metrics
     from sklearn.metrics import jaccard_score,accuracy_score, recall_score, fowlkes_mallows_score

     # initialize a spare improbability drive
     res = {}
     res['model'] = model.name
    
     # class predictions 
     if preds:
         y_true = y.flatten()
         y_pred = model.predict_classes(X).flatten()
         res['preds'] = [y_pred]

     if summary:
         summary = model.summary()
         res['summary'] = model.summary

     # FUSION MATRIX
     if fusion:
         if classes is None:
             classes=['0','1']
         else:
             classes=classes
         # Plot fusion matrix
         FM = fusion_matrix(matrix=(y_true,y_pred), 
                                     classes=classes)
         res['FM'] = FM

     # ROC Area Under Curve
     if roc:
         ROC = roc_plots(X, y, model)
         res['ROC'] = ROC

     # CLASSIFICATION REPORT
     if report:
         num_dashes=20
         print('\n')
         print('---'*num_dashes)
         print('\tCLASSIFICATION REPORT:')
         print('---'*num_dashes)
         # generate report
         report = metrics.classification_report(y_true,y_pred)
         res['report'] = report
         print(report)


     # save to dict:
     res['jaccard'] = jaccard_score(y_true, y_pred)
     res['fowlkes'] = fowlkes_mallows_score(y_true,y_pred)
     res['accuracy'] = accuracy_score(y_true, y_pred)
     res['recall'] = recall_score(y_true, y_pred)
    
     #Plot Model Training Results (PLOT KERAS HISTORY)
     if hist is not None:
         HIST = keras_history(hist)
         res['HIST'] = HIST

     return res
```


```python
res_m2 = computer.compute(X=X_test, y=y_test, model=m2, hist=h2, preds=True, 
               summary=False, report=True, fusion=True, 
               classes=['No Planet','Planet'],roc=True)
```

<img src="/assets/images/starskope/output_96_0.png" width=400>



<img src="/assets/images/starskope/output_96_1.png" width=400>

    
    
    ------------------------------------------------------------
    	CLASSIFICATION REPORT:
    ------------------------------------------------------------
                  precision    recall  f1-score   support
    
             0.0       1.00      0.98      0.99       565
             1.0       0.36      1.00      0.53         5
    
        accuracy                           0.98       570
       macro avg       0.68      0.99      0.76       570
    weighted avg       0.99      0.98      0.99       570
    



<img src="/assets/images/starskope/output_96_3.png" width=400>


```python
# We can retrieve a given metric from the computer's compute function 
# by accessing any of the given keys from the RES dictionary

# res_m2.keys()

# ['model']
# ['preds']
# ['summary']()
# ['FM'][0]
# ['FM'][1]
# ['ROC']
#print(res_m2['report'])
# ['jaccard']
 #res_m2['fowlkes']
# ['accuracy']
# ['recall']
# ['HIST']
```


```python
res_m2['fowlkes']
```




    0.9840290257993071



# `MODEL 3`

## Build M3


```python
#### MODEL 3
# increase learning rate to 4e-3

m3 = K.build_cnn(X_train, X_test, y_train, y_test, kernel_size=11, 
                     activation='relu', input_shape=X_train.shape[1:], 
                     strides=4, optimizer=Adam, learning_rate=4e-3, 
                     loss='binary_crossentropy', metrics=['accuracy'])
```

    BUILDING MODEL...
    LAYER 1
    LAYER 2
    LAYER 3
    LAYER 4
    FULL CONNECTION
    ADDING COST FUNCTION
    COMPILED


## Train M3


```python
# keep number of epochs at 20

h3 = K.fit_cnn(X_train,y_train, X_test, y_test, m3,
               validation_data=(X_test, y_test), verbose=2, epochs=20, 
               steps_per_epoch=(X_train.shape[1]//32), batch_size=32)
```

    FITTING MODEL...
    Epoch 1/20
     - 22s - loss: 0.6099 - accuracy: 0.6657 - val_loss: 0.0556 - val_accuracy: 0.9895
    Epoch 2/20
     - 21s - loss: 0.4555 - accuracy: 0.7945 - val_loss: 0.2497 - val_accuracy: 0.9088
    Epoch 3/20
     - 18s - loss: 0.3589 - accuracy: 0.8523 - val_loss: 0.3350 - val_accuracy: 0.8404
    Epoch 4/20
     - 17s - loss: 0.2505 - accuracy: 0.8987 - val_loss: 0.3346 - val_accuracy: 0.8649
    Epoch 5/20
     - 17s - loss: 0.3538 - accuracy: 0.8475 - val_loss: 0.2626 - val_accuracy: 0.8649
    Epoch 6/20
     - 17s - loss: 0.2716 - accuracy: 0.8911 - val_loss: 0.0878 - val_accuracy: 0.9737
    Epoch 7/20
     - 18s - loss: 0.2058 - accuracy: 0.9205 - val_loss: 0.1999 - val_accuracy: 0.9333
    Epoch 8/20
     - 17s - loss: 0.1560 - accuracy: 0.9441 - val_loss: 0.3655 - val_accuracy: 0.8333
    Epoch 9/20
     - 19s - loss: 0.1411 - accuracy: 0.9454 - val_loss: 0.2266 - val_accuracy: 0.8930
    Epoch 10/20
     - 23s - loss: 0.1298 - accuracy: 0.9504 - val_loss: 0.1879 - val_accuracy: 0.9456
    Epoch 11/20
     - 18s - loss: 0.1369 - accuracy: 0.9539 - val_loss: 0.0602 - val_accuracy: 0.9825
    Epoch 12/20
     - 22s - loss: 0.1084 - accuracy: 0.9618 - val_loss: 0.0592 - val_accuracy: 0.9807
    Epoch 13/20
     - 22s - loss: 0.0720 - accuracy: 0.9751 - val_loss: 0.0496 - val_accuracy: 0.9912
    Epoch 14/20
     - 18s - loss: 0.0986 - accuracy: 0.9640 - val_loss: 0.0982 - val_accuracy: 0.9719
    Epoch 15/20
     - 21s - loss: 0.1033 - accuracy: 0.9624 - val_loss: 0.0443 - val_accuracy: 0.9807
    Epoch 16/20
     - 19s - loss: 0.1129 - accuracy: 0.9586 - val_loss: 0.3545 - val_accuracy: 0.8596
    Epoch 17/20
     - 19s - loss: 0.1059 - accuracy: 0.9612 - val_loss: 0.4088 - val_accuracy: 0.8754
    Epoch 18/20
     - 17s - loss: 0.1018 - accuracy: 0.9634 - val_loss: 0.1042 - val_accuracy: 0.9614
    Epoch 19/20
     - 17s - loss: 0.0753 - accuracy: 0.9729 - val_loss: 0.0507 - val_accuracy: 0.9842
    Epoch 20/20
     - 17s - loss: 0.0587 - accuracy: 0.9779 - val_loss: 0.0699 - val_accuracy: 0.9789
    TRAINING COMPLETE
    Model: "sequential_3"
    _________________________________________________________________
    Layer (type)                 output Shape              Param #   
    =================================================================
    conv1d_9 (Conv1D)            (None, 3187, 8)           184       
    _________________________________________________________________
    max_pooling1d_9 (MaxPooling1 (None, 797, 8)            0         
    _________________________________________________________________
    batch_normalization_7 (Batch (None, 797, 8)            32        
    _________________________________________________________________
    conv1d_10 (Conv1D)           (None, 787, 16)           1424      
    _________________________________________________________________
    max_pooling1d_10 (MaxPooling (None, 197, 16)           0         
    _________________________________________________________________
    batch_normalization_8 (Batch (None, 197, 16)           64        
    _________________________________________________________________
    conv1d_11 (Conv1D)           (None, 187, 32)           5664      
    _________________________________________________________________
    max_pooling1d_11 (MaxPooling (None, 47, 32)            0         
    _________________________________________________________________
    batch_normalization_9 (Batch (None, 47, 32)            128       
    _________________________________________________________________
    conv1d_12 (Conv1D)           (None, 37, 64)            22592     
    _________________________________________________________________
    max_pooling1d_12 (MaxPooling (None, 9, 64)             0         
    _________________________________________________________________
    flatten_3 (Flatten)          (None, 576)               0         
    _________________________________________________________________
    dropout_5 (Dropout)          (None, 576)               0         
    _________________________________________________________________
    dense_7 (Dense)              (None, 64)                36928     
    _________________________________________________________________
    dropout_6 (Dropout)          (None, 64)                0         
    _________________________________________________________________
    dense_8 (Dense)              (None, 64)                4160      
    _________________________________________________________________
    dense_9 (Dense)              (None, 1)                 65        
    =================================================================
    Total params: 71,241
    Trainable params: 71,129
    Non-trainable params: 112
    _________________________________________________________________


## Evaluate M3


```python
res_m3 = computer.compute(X=X_test, y=y_test, model=m3, hist=h3, preds=True, 
               summary=False, report=True, fusion=True, 
               classes=['No Planet','Planet'],roc=True)
```


<img src="/assets/images/starskope/output_105_0.png" width=400>

<img src="/assets/images/starskope/output_105_1.png" width=400>


    
    
    ------------------------------------------------------------
    	CLASSIFICATION REPORT:
    ------------------------------------------------------------
                  precision    recall  f1-score   support
    
             0.0       1.00      0.98      0.99       565
             1.0       0.29      1.00      0.45         5
    
        accuracy                           0.98       570
       macro avg       0.65      0.99      0.72       570
    weighted avg       0.99      0.98      0.98       570
    


<img src="/assets/images/starskope/output_105_3.png" width=400>


It appears that increasing the learning rate did not help to improve the model's performance. While it still identified all 5 planets, it misclassified 4 non-planets as planets, two more than model 2. Let's see if we can decrease the False Positive Rate, while Maintaining the False Negative Rate at zero. We'll go back to Model 2's original learning rate of 3e-4, this time increasing the number of epochs instead.

# `MODEL 4`

## Build M4


```python
#### MODEL 4
# decrease learning rate back to 3e-4

m4 = K.build_cnn(X_train, X_test, y_train, y_test, kernel_size=11, 
                     activation='relu', input_shape=X_train.shape[1:], 
                     strides=4, optimizer=Adam, learning_rate=3e-4, 
                     loss='binary_crossentropy', metrics=['accuracy'])
```

    BUILDING MODEL...
    LAYER 1
    LAYER 2
    LAYER 3
    LAYER 4
    FULL CONNECTION
    ADDING COST FUNCTION
    COMPILED


## Fit M4


```python
# increase number of epochs to 33

h4 = K.fit_cnn(X_train,y_train, X_test, y_test, m4,
               validation_data=(X_test, y_test), verbose=2, epochs=33, 
               steps_per_epoch=(X_train.shape[1]//32), batch_size=32)
```

    FITTING MODEL...
    Epoch 1/33
     - 17s - loss: 0.6739 - accuracy: 0.5931 - val_loss: 0.8185 - val_accuracy: 0.1211
    Epoch 2/33
     - 18s - loss: 0.6006 - accuracy: 0.6777 - val_loss: 0.6971 - val_accuracy: 0.4772
    Epoch 3/33
     - 20s - loss: 0.5806 - accuracy: 0.6989 - val_loss: 0.4942 - val_accuracy: 0.8281
    Epoch 4/33
     - 19s - loss: 0.5300 - accuracy: 0.7412 - val_loss: 0.4063 - val_accuracy: 0.8439
    Epoch 5/33
     - 18s - loss: 0.4974 - accuracy: 0.7620 - val_loss: 0.4653 - val_accuracy: 0.7614
    Epoch 6/33
     - 18s - loss: 0.4563 - accuracy: 0.7907 - val_loss: 0.7816 - val_accuracy: 0.5632
    Epoch 7/33
     - 18s - loss: 0.4144 - accuracy: 0.8239 - val_loss: 0.5558 - val_accuracy: 0.7140
    Epoch 8/33
     - 18s - loss: 0.3503 - accuracy: 0.8561 - val_loss: 0.1770 - val_accuracy: 0.9228
    Epoch 9/33
     - 19s - loss: 0.3230 - accuracy: 0.8636 - val_loss: 0.2454 - val_accuracy: 0.8807
    Epoch 10/33
     - 18s - loss: 0.2673 - accuracy: 0.8984 - val_loss: 0.9147 - val_accuracy: 0.6526
    Epoch 11/33
     - 18s - loss: 0.2434 - accuracy: 0.9091 - val_loss: 0.8955 - val_accuracy: 0.6228
    Epoch 12/33
     - 18s - loss: 0.2401 - accuracy: 0.9119 - val_loss: 0.5549 - val_accuracy: 0.7474
    Epoch 13/33
     - 17s - loss: 0.1985 - accuracy: 0.9227 - val_loss: 0.2674 - val_accuracy: 0.8860
    Epoch 14/33
     - 19s - loss: 0.2019 - accuracy: 0.9255 - val_loss: 0.1316 - val_accuracy: 0.9439
    Epoch 15/33
     - 17s - loss: 0.1642 - accuracy: 0.9416 - val_loss: 0.0758 - val_accuracy: 0.9702
    Epoch 16/33
     - 17s - loss: 0.1833 - accuracy: 0.9331 - val_loss: 0.0558 - val_accuracy: 0.9860
    Epoch 17/33
     - 17s - loss: 0.1506 - accuracy: 0.9476 - val_loss: 0.2535 - val_accuracy: 0.9035
    Epoch 18/33
     - 17s - loss: 0.1445 - accuracy: 0.9501 - val_loss: 0.1732 - val_accuracy: 0.9368
    Epoch 19/33
     - 17s - loss: 0.1302 - accuracy: 0.9558 - val_loss: 0.1816 - val_accuracy: 0.9474
    Epoch 20/33
     - 17s - loss: 0.1176 - accuracy: 0.9586 - val_loss: 0.0928 - val_accuracy: 0.9737
    Epoch 21/33
     - 17s - loss: 0.1099 - accuracy: 0.9653 - val_loss: 0.0820 - val_accuracy: 0.9754
    Epoch 22/33
     - 17s - loss: 0.1263 - accuracy: 0.9552 - val_loss: 0.0566 - val_accuracy: 0.9825
    Epoch 23/33
     - 18s - loss: 0.1067 - accuracy: 0.9628 - val_loss: 0.0553 - val_accuracy: 0.9877
    Epoch 24/33
     - 19s - loss: 0.1192 - accuracy: 0.9605 - val_loss: 0.0911 - val_accuracy: 0.9649
    Epoch 25/33
     - 17s - loss: 0.0929 - accuracy: 0.9675 - val_loss: 0.1150 - val_accuracy: 0.9632
    Epoch 26/33
     - 18s - loss: 0.0912 - accuracy: 0.9688 - val_loss: 0.0455 - val_accuracy: 0.9895
    Epoch 27/33
     - 19s - loss: 0.0942 - accuracy: 0.9672 - val_loss: 0.0470 - val_accuracy: 0.9877
    Epoch 28/33
     - 18s - loss: 0.0875 - accuracy: 0.9700 - val_loss: 0.0856 - val_accuracy: 0.9737
    Epoch 29/33
     - 18s - loss: 0.0784 - accuracy: 0.9713 - val_loss: 0.0888 - val_accuracy: 0.9754
    Epoch 30/33
     - 18s - loss: 0.0860 - accuracy: 0.9706 - val_loss: 0.0523 - val_accuracy: 0.9895
    Epoch 31/33
     - 18s - loss: 0.0967 - accuracy: 0.9694 - val_loss: 0.1047 - val_accuracy: 0.9649
    Epoch 32/33
     - 18s - loss: 0.0722 - accuracy: 0.9757 - val_loss: 0.0571 - val_accuracy: 0.9825
    Epoch 33/33
     - 17s - loss: 0.0965 - accuracy: 0.9678 - val_loss: 0.1138 - val_accuracy: 0.9596
    TRAINING COMPLETE
    Model: "sequential_5"
    _________________________________________________________________
    Layer (type)                 output Shape              Param #   
    =================================================================
    conv1d_17 (Conv1D)           (None, 3187, 8)           184       
    _________________________________________________________________
    max_pooling1d_17 (MaxPooling (None, 797, 8)            0         
    _________________________________________________________________
    batch_normalization_13 (Batc (None, 797, 8)            32        
    _________________________________________________________________
    conv1d_18 (Conv1D)           (None, 787, 16)           1424      
    _________________________________________________________________
    max_pooling1d_18 (MaxPooling (None, 197, 16)           0         
    _________________________________________________________________
    batch_normalization_14 (Batc (None, 197, 16)           64        
    _________________________________________________________________
    conv1d_19 (Conv1D)           (None, 187, 32)           5664      
    _________________________________________________________________
    max_pooling1d_19 (MaxPooling (None, 47, 32)            0         
    _________________________________________________________________
    batch_normalization_15 (Batc (None, 47, 32)            128       
    _________________________________________________________________
    conv1d_20 (Conv1D)           (None, 37, 64)            22592     
    _________________________________________________________________
    max_pooling1d_20 (MaxPooling (None, 9, 64)             0         
    _________________________________________________________________
    flatten_5 (Flatten)          (None, 576)               0         
    _________________________________________________________________
    dropout_9 (Dropout)          (None, 576)               0         
    _________________________________________________________________
    dense_13 (Dense)             (None, 64)                36928     
    _________________________________________________________________
    dropout_10 (Dropout)         (None, 64)                0         
    _________________________________________________________________
    dense_14 (Dense)             (None, 64)                4160      
    _________________________________________________________________
    dense_15 (Dense)             (None, 1)                 65        
    =================================================================
    Total params: 71,241
    Trainable params: 71,129
    Non-trainable params: 112
    _________________________________________________________________


## Evaluate M4


```python
res_m4 = computer.compute(X=X_test, y=y_test, model=m4, hist=h4, preds=True, 
               summary=False, report=True, fusion=True, 
               classes=['No Planet','Planet'],roc=True)
```

<img src="/assets/images/starskope/output_113_0.png" width=400>


<img src="/assets/images/starskope/output_113_1.png" width=400>


    
    
    ------------------------------------------------------------
    	CLASSIFICATION REPORT:
    ------------------------------------------------------------
                  precision    recall  f1-score   support
    
             0.0       1.00      0.96      0.98       565
             1.0       0.18      1.00      0.30         5
    
        accuracy                           0.96       570
       macro avg       0.59      0.98      0.64       570
    weighted avg       0.99      0.96      0.97       570
    



<img src="/assets/images/starskope/output_113_3.png" width=400>

# Interpret Results

## Conclusion

Above, we were able to identify with 99% accuracy 5 of 5 stars with an orbiting exoplanet (or exoplanets). The best model (MODEL 2) incorrectly predicted just 2 False Positives, with ZERO false negatives. 

## Save Weights (Model 2)


```python
# # %mkdir models
# m2.save_weights('models/k2_cnn1d.h5')
```

# Recommendations

While it is possible to create a near-perfect classification model for detecting exoplanets using the raw flux values of an imbalanced data set, the model would benefit from further validation using additional data from either K2 or another telescope such as TESS. One issue with this model is that it doesn't reveal how it makes the decision on each prediction, an insight that would be extremely useful for astrophysicists and for developing and improving the model itself. A better, more robust and useful model, therefore, would be one which gives such insight without sacrificing accuracy or recall. 

My recommendations are the following:

1. Use datasets from the MAST website (via API) to incorporate other calculations of the star's properties as features to be used for classification algorithms. Furthermore, attempt other types of transformations and normalizations on the data before running the model - for instance, apply Fourier transform and phase folding.

2. Combine data from multiple campaigns and perhaps even multiple telescopes (for instance, matching sky coordinates and time intervals between K2, Kepler, and TESS for a batch of stars that have overlapping observations - this would be critical for finding transit periods that are longer than the campaigns of a single telecope's observation period).

3. Explore using computer vision on not only the Full Frame images we can collect from telescopes like TESS, but also on spectographs of the flux values themselves. The beauty of machine learning is our ability to rely on the computer to pick up very small nuances in differences that we ourselves cannot see with our own eyes. 
   
4. Explore using autoencoded machine learning algorithms with Restricted Boltzmann Machines - this type of model has proven to be incredibly effective in the image analysis of handwriting as we've seen applied the MNIST dataset - let's find out if the same is true for images of stars, be they the Full Frame Images or spectographs.

# Future Work

To continue this project, I'll take another approach for detecting exoplanets using computer vision to analyze images of spectographs of this same star flux data set. In part II [starskøpe-2 I use Restricted Boltzmann Machines on Fourier-transformed spectograph images of the Flux data for K2. These are then stacked on top of each other as layers in a Deep Boltzmann Machine neural network. In part III (notebook `[starskøpe-3]`) I will apply a similar technique using data from TESS.

For triage/vetting purposes, this model could be useful for scientists. However, I would like to extend the model's capability using a multiclassification algorithm that can tell us not only if there is a transiting body, but how many, as well as additional information about the star's properties. The latter could be done by training another model on stellar parameters, and then stacking the two together into one neural network.
