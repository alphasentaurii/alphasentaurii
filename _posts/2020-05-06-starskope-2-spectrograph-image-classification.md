---
layout: post
title:  "Spectrograph Image Classification"
date:   2020-05-06 11:11:11 -1111
categories: datascience
---

For the next phase of the Starskøpe planet hunting project, I used `Google Colabs` to generate spectrograph images of the same star flux signals dataset from Kaggle. Due to memory constraints, I started out by only using a portion of this already small dataset as a test round. To save time, I ran the images through a `Keras Convolutional Network` using Theano, similar to the one I built in the first phase of the project. The results were not ideal, but this was to be expected. 

 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZRUudjsf0ofOMVjhzfF239Df3BaO2jqp#scrollTo=n0aJxtzZMCSI&uniqifier=4)

## Phase 3: Scrape MAST via API on AWS

In order to improve model, therefore, the next step was to generate a larger dataset of images, using not only all the data from Kaggle, but also import additional signal datasets from the MAST API. Traditionally, this can be done from the MAST website directly. However, it is now possible to scrape the data for free from AWS where it is being (at least temporarily) housed. 



