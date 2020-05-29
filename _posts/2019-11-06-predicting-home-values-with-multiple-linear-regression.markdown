---
layout: post
title:  "Predicting Home Values with Multiple Linear Regression"
date:   2019-11-06 10:23:47 -0800
categories: datascience
---


project links:

> [`slides`](/projects/king-county/slides/index.html)

> [`project repo`](https://github.com/hakkeray/predicting-home-values-with-multiple-linear-regression)

## PROJECT GOAL
Identify best combination of variables for predicting property values (house prices) in King County, Washington, USA.

## INTRODUCTION
Ask any realtor what are the top 3 most important variables for measuring property value, and they will all say the same thing: 1) location, 2) location, and 3) location. I asked a friend who has been doing real estate for about 20 years (we'll call her "Mom") what other factors besides location tend to have some impact and she mentioned the following:

1. **Square-Footage** almost every time, a bigger house is going to cost more than a smaller one.

2. **Condition** of the house ("Is it a fixer-upper?")

3. **# Bathrooms** she specifically said number of bathrooms outweighs number of bedrooms (although sometimes more bathrooms will mean more bedrooms, that's not always the case.)

This project is going to tell us mathematically if her assumptions are valid or not.

## ASSUMPTIONS
Couple of other assumptions to consider for this analysis:

1. **Market Demand** Mom also mentioned how market demand changes from generation to generation. For example, right now (Nov 2019) more and more 'millenials' are buying houses, but unlike their parents who might be more inclined toward buying a property with a lot of land sitting farther away from the city, millenials generally want the opposite. They want to be close to the hustle and bustle, they want "fixer-uppers" they can buy at a lower price and spend their money making it their own.

2. **Non-Universality** The selling-factors for real estate in one town are not necessarily going to hold true for a town on the other side of the country. In other words, we can't automatically assume the predictors we identify in this dataset are universal.

## THE CLIENT
When it comes to real estate, or selling anything for that matter, it's absolutely critical to keep in mind what is going on in the market, what your target demographic is, and most of all, what do they want? For this project, the client is someone who "flips" houses and is looking to buy property in the Greater Seattle area. Our job is to help them identify which factors are most important to consider before they purchase a house for flipping (i.e. reselling for a higher price).

---

### THE DATA

Starting with over 25,000 data points spread across 18 possible variables, I narrowed the candidates down to just three predictors of price. I'll briefly walk you through how and why I eliminated the other 15:

DATAFRAME COLUMNS:
- id
- date
- price

- waterfront
- view

- yr_built
- yr_renovated
- condition
- grade

- zipcode
- lat
- long

- bedrooms
- bathrooms
- floors

- sqft_above
- sqft_basement
- sqft_living
- sqft_lot
- sqft_living15
- sqft_lot15

