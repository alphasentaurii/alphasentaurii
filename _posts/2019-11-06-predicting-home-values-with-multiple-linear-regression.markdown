---
layout: post
title:  "Predicting Home Values with Multiple Linear Regression"
date:   2019-11-06 10:23:47 -0800
categories: projects datascience
---

# Predicting Home Values with Multiple Linear Regression

project links:
> [`jupyter notebook`](./projects/king-county/notebook.html)
> [`slides`](./projects/king-county/slides/index.html)
> [`functions`](./code.html)
> [`project repo`](./)

Table of Contents


The goal of this project was to identify best combination of variable(s) for predicting property values in King County, Washington, USA.

## DOMAIN KNOWLEDGE
Ask any realtor you'll ever meet what the top three most important variables are for measuring property value, and they will all tell you the same 3 things: 1) location, 2) location, and 3) location. After that opinions may vary, but I asked a friend who has been doing real estate for about 20 years (Mom) what other factors tend to have some impact and she mentioned the following:

* square-footage (almost every time, a bigger house is going to cost more than a smaller one)

* condition of the house ("is it a fixer-upper?")

* how many bathrooms. Interestingly enough, she specifically pointed out that the number of bathrooms is more important than bedrooms to most people, although usually more bathrooms means more bedrooms, that's not always the case.

Couple of other notes to bear in mind about assumptions to factor in to our analysis:

* market demand. Mom also mentioned how market demand changes from generation to generation. For example, right now (Nov 2019) more and more 'millenials' are buying houses, but unlike their parents who might be more inclined toward buying a property with a lot of land, a little ways away from the noisy and busy city, millenials essentially want the opposite. They want to be close to the city where all the "action" is, they want "fixer-uppers" they can buy at a lower market value and spend their money making it their own.

* Extending this further, the selling-factors for real estate in one town is not necessarily going to be the same on, say, the other side of the country. In other words, we can't automatically assume the predictors we identify in this dataset are universal.

* In conclusion, when it comes to real estate, or selling anything for that matter, it's absolutely critical to keep in mind what is going on in the market, what your target demographic is, and most of all, what do they want? (Full disclosure: I've worked in marketing for almost a decade now and a couple of years ago co-founded a small marketing agency. So, those questions are always in the back of my mind if the question is about how to sell something. Several customers were real estate companies who hired us to produce marketing videos for their properties as well as design their websites. So without further ado, let's look at the actual math and see if the assumptions laid out above actually check out.
