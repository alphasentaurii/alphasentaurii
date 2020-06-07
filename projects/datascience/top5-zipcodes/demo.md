---
layout: page
title:  "Top Five Zip Codes"
---

# Top 5 Zip Codes Project Demo

Top Five Zip Codes is a `housing market prediction model` that uses `seasonal ARIMA time-series analysis and GridsearchCV` to `recommend the top 5 zip codes for purchasing a single-family home in Westchester, New York`.

<div style="display:block; text-align:center; clear:both; position:relative; z-index:9999;"><iframe src="https://player.vimeo.com/video/384921005" width="640" height="480" frameborder="0" allow="autoplay; fullscreen" allowfullscreen></iframe>
</div>

## Business Case

Recommend top 5 zipcodes for client interested in buying a single-family home in Westchester County, NY within the next two years.

## Required Parameters (from Client)

1. Timeframe: Purchase of new home would take place within the next two years
2. Budget: Cost to buy not to exceed `800,000 USD` maximum
3. Location: Zip Code search radius restricted to Westchester County, New York

## Ideal Parameters (from Client)

* Commute: Towns/Villages having shortest/fastest commute New York City
* School District: Zip codes include towns with A/A+ school district rating

## Success Criteria for Model

1. Maximum ROI (highest forecasted home value increase for lowest upfront cost)
2. Confidence Intervals
3. Risk mitigation: financial stability of homeowners and homes based on historic data

## Objective

Make forecast predictions for zip codes and their mean home values using Seasonal ARIMA time-series analysis.

## Commute Times

Since commute time to Grand Central Station is part of the client's required criteria, I first had to look up which towns/zip codes were on which train lines. Grand Central has 3 main lines on Metro North Railroad: Hudson, Harlem, and New Haven. The first question I was interested in answering was if the average home prices for the zip codes that fall under these geographic sections display any trends. 

### Mean Values by Train Line

<div style="width:400px">
<img class="img-responsive" src="http://hakkeray.com/assets/images/timeseries/meanvalues_area.png"></div>

### New Haven line

<div style="width:400px">
<img class="img-responsive" src="http://hakkeray.com/assets/images/timeseries/newhaven_mapTime.png"></div>

Note that this does not include zip codes in Connecticut (which the New Haven line covers) since the client is only interested in towns in New York state. 

### Harlem Line

<div style="width:400px">
<img class="img-responsive" src="http://hakkeray.com/assets/images/timeseries/harlem_mapTime.png"></div>

### Hudson Line

<div style="width:400px">
<img class="img-responsive" src="http://hakkeray.com/assets/images/timeseries/hudson_mapTime.png"></div>

## GridsearchCV with SARIMAX

I then ran a gridsearch using a Seasonal ARIMA (SARIMAX) model to make forecast predictions on all 61 zip codes in Westchester County. Using PANDAS, I narrowed down the list by top 10 highest ROI zip codes. I then identified which of these had the lowest confidence intervals in order to ensure I was only selecting the most accurate results.

<div style="width:400px">
<img class="img-responsive" src="http://hakkeray.com/assets/images/timeseries/conf_roi_pred_3D.png"></div>

<div style="width:400px">
<img class="img-responsive" src="http://hakkeray.com/assets/images/timeseries/conf_roi_heatmap.png"></div>

## TOP FIVE ZIP CODES

The top five I selected based on the above criteria were 10549, 10573, 10604, 10605, 10706:

<div style="width:400px">
<img class="img-responsive" src="http://hakkeray.com/assets/images/timeseries/10549.png"></div>

<div style="width:400px">
<img class="img-responsive" src="http://hakkeray.com/assets/images/timeseries/10573.png"></div>

<div style="width:400px">
<img class="img-responsive" src="http://hakkeray.com/assets/images/timeseries/10604.png"></div>

<div style="width:400px">
<img class="img-responsive" src="http://hakkeray.com/assets/images/timeseries/10605.png"></div>

<div style="width:400px">
<img class="img-responsive" src="http://hakkeray.com/assets/images/timeseries/10706.png"></div>

<div style="width:400px">
<img class="img-responsive" src="http://hakkeray.com/assets/images/timeseries/top5_final_mapTime.png"></div>

## Contact

You can reach me at <rukeine@gmail.com>.

## License

This project uses the following license: [MIT License](/LICENSE.md).
