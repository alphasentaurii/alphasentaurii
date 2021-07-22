---
layout: page
title:  "Predicting Home Values Project Demo"
---

## PROJECT DEMO: Predicting Home Values

The goal of this data science project was to identify a strategy for increasing the sale price or property value of a home in King County, Washington.

<!-- <div style="display:block; text-align:center; clear:both; position:relative; z-index:9999;">
<iframe src="https://player.vimeo.com/video/371786438" width="640" height="360" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen sandbox="allow-scripts"></iframe>
    <br /><br />
</div> -->

## Dataset

The dataset is comprised of over 20,000 home sales between May 2014 - May 2015. In addition to the price at time of sale, each observation (or row) was associated with 18 dependent variables (or `features`). For my analysis, I was interested in identifying **which combination of variables** are the best predictors of high property values. Hence the term _multiple_ linear regression (a simple linear regression model would inaccurately only account for one factor. 

## Features

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

The model eliminated most of these features, leaving me with just 3: square footage of the home's living area, the graded score of the property, and the zip code as well as latitude and longitude where the property was located. Let's take a closer look at these.

### 1. Square-footage (Living area) Increases with Price

This includes the square footage of the basement, but excludes square footage of the land around the home. Not surprisingly, it turns out that homes with a higher square footage of living space tend to have much higher property values. The bigger the house, the higher the sale price. You can see from the scatter plot below that as the property values increase upward, the square-footage also increases to the right.

#### Square-Foot Living (Scatterplot)

<div style="background-color:white">
<img src="/assets/images/king-county/sqft-living-scatterplot.png" alt="sqft living scatterplot" title="Sqft Living Scatterplot" width="400"/>
</div>

### 2. Grade

Next, I looked at `GRADE`. Each property in the dataset is associated with a score ranging from 1 to 13, 13 being the best. The factors that actually determine that score have to do with the quality of materials in the home, the wood, marble, etc., as well as the quality of carpentry and craftsmanship put into those materials. Much like the scatterplot we saw before, the box plot (below) shows how higher scores in Grade lead to higher property values. 

#### Grade (Boxplot) Increases with Price

<div style="background-color:white">
<img src="/assets/images/king-county/grade-boxplot.png" alt="grade boxplot" title="Grade Boxplot" width="400"/>
</div>

#### High Grade Scores Less Common 

In the plot below, it's also noticeable that there are far fewer homes that achieve a grade of 13, indicating these homes might be unique in some way, perhaps the architect is well-known, or the materials themselves are rare, all of which factors into the overall grade of the property regardless of how many bedrooms or square-footage of the land, etc. 

#### Distribution of Grade (scatterplot)

<div style="background-color:white">
<img src="/assets/images/king-county/grade-scatterplot.png" alt="grade scatterplot" title="Grade Scatterplot" width="400"/></div>

#### 3. Location, Location, Location

Now, don't take it just from me - if you ask any realtor what are the three most important factors for selling your home at the highest possible value, they'll all say the same thing: Location, Location, and Location. What I can tell you is that this is in fact mathematically true, according the model: median home values increase or decrease depending on the `zip code`.

### 3. Location + Grade ?

I was curious whether there is any relationship between Grade and Location, so I created this plot by mapping each home's latitude and longitude, then I applied a color scheme to see if certain grades happen to fall in any kind of geographic pattern. Sure enough, you can see that the pink and purple dots, which are the highest grade scores, are far fewer, and they tend to clump around a specific area, which just so happens to be - Seattle - a city known for its prime real estate. However, according the model, you need to have a higher grade as well as a specific location in order to capture te highest possible price. In other words, both features are critical and you can't simply increase the price of a home simply by having one feature without the other. You need both. 

#### Latitude and Longitude: Geographic Distribution of Grade

<div style="background-color:white">
<img src="/assets/images/king-county/grade-lat-long.png" alt="grade lat long" title="Grade and Location" width="400"/>
</div>

#### Median Home Values by Zip Code 

So just to show you how that breaks down, below is a map of King County where the lines indicate the zip code boundaries. The lighter shades of purple indicate lower property values, whereas the darkest shades of purple represent the highest property values. The darkest one on here is, of course, Seattle.

#### Interactive Map

<iframe src="https://public.tableau.com/views/HousePricesbyZipCodeinKingCountyWA/KingCounty?:display_count=y&publish=yes&:origin=viz_share_link" width=600 height=600 webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>


#### Look Up Median Home Values by Zip Code

Now let's say you want to know which zip codes fall under a given range of home values. Maybe you only want to consider zip codes where the median value is $1,000,000 or higher. So I wrote a function for doing just that, and for King County, the million dollar question gives us only one zip code, 98039, and if we drop it down to say half a million for median home values, we get back 14 zip codes to choose from.

## Conclusion

According to our final model, the best predictors of house prices are sqft-living, zipcode, and grade.  What's really important to keep in mind is that the model eliminated 15 possible variables from having a significant impact on price - so what that means is if you have a home in one of these zip codes, it doesn't matter so much how many bedrooms or bathrooms you have, the mere fact that your property is in one of these zip codes automatically increases the property value. Beyond that, the only factors for increasing the home's value that you need to focus on would be the grade and the square footage of the living space.

## Summary

So to recap, increasing the price of your home comes down to three critical factors: the square footage of the living area, the grade (materials and craftsmanship), and the location (zip code). **Location is so important that no matter how much money you invest in building a bigger house, even with the most expensive and rare materials and a fancy architect, if it's not a "desirable" location, in this case, too far outside Seattle, you're not going to sell the house for the kind of price you'd otherwise get closer to the city.**

## Recommendations

1. Homes with larger living areas are valued higher than smaller homes.
2. Houses in certain zip codes are valued at higher prices than other zip codes.
3. Homes that score above at least 8 on Grade will sell higher than those below.

## Future Work

### 1. Do house prices change over time or depending on season? 

This data set was limited to a one-year time-frame. I'd be interested in widening the sample size to investigate how property values fluctuate over time as well as how they are affected by market fluctuations.

### 2. Resold Homes

Can we validate the accuracy of our prediction model by looking specifically at houses that resold for a higher price in a given timeframe? In other words, try to identify which specific variables changed (e.g. increased grade score after doing renovations) and therefore were determining factors in the increased price of the home when it was resold.

## Show Me The CODE

[`github repo`](https://github.com/hakkeray/predicting-home-values-with-multiple-linear-regression)

## CONTACT 
<a href="mailto:rukeine@gmail.com">rukeine@gmail.com</a>

## LICENSE
