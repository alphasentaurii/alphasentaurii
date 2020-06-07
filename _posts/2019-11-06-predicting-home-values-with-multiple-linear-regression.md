---
layout: post
title:  "Predicting Home Values"
date:   2019-11-06 10:23:47 -0800
categories: datascience
---

<iframe src="https://player.vimeo.com/video/371786438" width="640" height="360" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen sandbox="allow-scripts"></iframe>

## What are the best predictors of property value?

Ask any realtor what are the top 3 most important variables for measuring property value, and they will all say the same thing: 1) location, 2) location, and 3) location. I asked a friend who has been doing real estate for about 20 years (we'll call her "Mom") what other factors besides location tend to have some impact and she mentioned the following:

1. Square-footage
2. Condition
3. Number of Bathrooms
4. Market Demand
5. Geography
6. The Buyer

### 1. Square-Footage

Almost every time, a bigger house is going to cost more than a smaller one.

### 2. Condition

Condition of the house: "Is it a fixer-upper?"

### 3. Number of Bathrooms

She specifically said number of bathrooms outweighs number of bedrooms: "although sometimes more bathrooms will mean more bedrooms, that's not always the case". This project is going to tell us mathematically if her assumptions are valid or not.

## Market Demand

Mom also mentioned how market demand changes from generation to generation. For example, right now (Nov 2019) more and more 'millenials' are buying houses, but unlike their parents who might be more inclined toward buying a property with a lot of land sitting farther away from the city, millenials generally want the opposite. They want to be close to the hustle and bustle, they want "fixer-uppers" they can buy at a lower price and spend their money making it their own.

## Geographic Differences

The selling-factors for real estate in one town are not necessarily going to hold true for a town on the other side of the country. In other words, we can't automatically assume the predictors we identify in this dataset are universal.

## The Client
When it comes to real estate, or selling anything for that matter, it's absolutely critical to keep in mind what is going on in the market, what your target demographic is, and most of all, what do they want? For this project, the client is someone who "flips" houses and is looking to buy property in the Greater Seattle area. Our job is to help them identify which factors are most important to consider before they purchase a house for flipping (i.e. reselling for a higher price).

# Interactive Map: Property Values by Zip Code

<div style="width:600px">
<img class="img-responsive" src="http://hakkeray.com/assets/images/king-county/tableau-map-kingcounty.png"></div>

## Goal

The goal was to identify a strategy for increasing the sale price or property value of a given home in this location. To achieve the above goal, I used a `multiple linear regression model` to identify the best combination of variables for predicting the `target`: price of a home at the time of sale. 

## Dataset

The dataset is comprised of over 20,000 home sales between May 2014 - May 2015. In addition to the price at time of sale, each observation (or row) was associated with 18 dependent variables (or `features`). For my analysis, I was interested in identifying **which combination of variables** are the best predictors of high property values. Hence the term _multiple_ linear regression (a simple linear regression model would inaccurately only account for one factor. 

## Features

The model eliminated most of these features, leaving me with just 3: square footage of the home's living area, the graded score of the property, and the zip code as well as latitude and longitude where the property was located.

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

## Top 3 Features

Let's take a closer look at these "top 3" features.

### 1. Square-footage (Living area)

This includes the square footage of the basement, but excludes square footage of the land around the home. Not surprisingly, it turns out that homes with a higher square footage of living space tend to have much higher property values. The bigger the house, the higher the sale price. You can see from the scatter plot below that as the property values increase upward, the square-footage also increases to the right.

<div style="width:400px">
<img class="img-responsive" src="http://hakkeray.com/assets/images/king-county/sqft-living-scatterplot.png"></div>

### 2. Grade

Next, I looked at something called `GRADE`. Each property in the dataset is associated with a score ranging from 1 to 13, 13 being the best. The factors that actually determine that score have to do with the quality of materials in the home, the wood, marble, etc., as well as the quality of carpentry and craftsmanship put into those materials. Much like the scatterplot we saw before, the box plot (below) shows how higher scores in Grade lead to higher property values. 

<div style="width:400px">
<img class="img-responsive" src="http://hakkeray.com/assets/images/king-county/grade-boxplot.png"></div>

In the plot below, it's also noticeable that there are far fewer homes that achieve a grade of 13, indicating these homes might be unique in some way, perhaps the architect is well-known, or the materials themselves are rare, all of which factors into the overall grade of the property regardless of how many bedrooms or square-footage of the land, etc. 

<div style="width:400px">
<img class="img-responsive" src="http://hakkeray.com/assets/images/king-county/grade-scatterplot.png"></div>

# 3. Zip Code

Finally, one of the most important factors determing the property value is location. Now one thing I was curious about was whether there is any relationship between Grade and Location, so I created this plot by mapping each home's latitude and longitude, then I applied a color scheme to see if certain grades happen to fall in any kind of geographic pattern. Sure enough, you can see that the pink and purple dots, which are the highest grade scores, are far fewer, and they tend to clump around a specific area, which just so happens to be - Seattle - a city known for its prime real estate. However, according the model, you need to have a higher grade as well as a specific location in order to capture te highest possible price. In other words, both features are critical and you can't simply increase the price of a home simply by having one feature without the other. You need both. 

## Location, Location, Location

Now, don't take it just from me - if you ask any realtor what are the three most important factors for selling your home at the highest possible value, they'll all say the same thing: Location, Location, and Location. What I can tell you is that this is in fact mathematically true, according the model: median home values increase or decrease depending on the zip code.

### Map

So just to show you how that breaks down, below is a map of King County where the lines indicate the zip code boundaries. The lighter shades of purple indicate lower property values, whereas the darkest shades of purple represent the highest property values. The darkest one on here is, of course, Seattle.


# Interactive Map

<html>
<body>
<iframe src="https://public.tableau.com/views/HousePricesbyZipCodeinKingCountyWA/KingCounty?:display_count=y&publish=yes&:origin=viz_share_link" width=600 height=600 webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>
</body>
</html>

# Look Up Median Home Values by Zip Code

Now let's say you want to know which zip codes fall under a given range of home values. Maybe you only want to consider zip codes where the median value is $1,000,000 or higher. So I wrote a function for doing just that, and for King County, the million dollar question gives us only one zip code, 98039, and if we drop it down to say half a million for median home values, we get back 14 zip codes to choose from. 

# Conclusion

According to our final model, the best predictors of house prices are sqft-living, zipcode, and grade.

What's really important to keep in mind is that the model eliminated 15 possible variables from having a significant impact on price - so what that means is if you have a home in one of these zip codes, it doesn't matter so much how many bedrooms or bathrooms you have, the mere fact that your property is in one of these zip codes automatically increases the property value. Beyond that, the only factors for increasing the home's value that you need to focus on would be the grade and the square footage of the living space.

# Recommendations

So to recap, increasing the price of your home comes down to three critical factors: the square footage of the living area, the grade, which again comes down to materials and craftsmanship, and the location. However location is so important that no matter how much money you invest in building a bigger house, even with the most expensive and rare materials and a fancy architect, if it's not a "desirable" location, in this case, too far outside Seattle, you're not going to sell the house for the kind of price you'd otherwise get closer to the city.

1. Homes with larger living areas are valued higher than smaller homes.
2. Houses in certain zip codes are valued at higher prices than other zip codes.
3. Homes that score above at least 8 on Grade will sell higher than those below.

# Future Work

## 1. Do house prices change over time or depending on season? 

This data set was limited to a one-year time-frame. I'd be interested in widening the sample size to investigate how property values fluctuate over time as well as how they are affected by market fluctuations.

## 2. Resold Homes

Can we validate the accuracy of our prediction model by looking specifically at houses that resold for a higher price in a given timeframe? In other words, try to identify which specific variables changed (e.g. increased grade score after doing renovations) and therefore were determining factors in the increased price of the home when it was resold.

[`github repo`](https://github.com/hakkeray/predicting-home-values-with-multiple-linear-regression)