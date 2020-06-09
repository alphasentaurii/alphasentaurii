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

<div style="background-color:white">
<img src="/assets/images/timeseries/meanvalues_area.png" alt="Mean Values by Train Line Area Plot" title="Mean Values by Train Line Area Plot" width="400"/>
</div>

### New Haven line

<div style="background-color:white">
<img src="/assets/images/timeseries/newhaven_mapTime.png" alt="New Haven Line Zip Code Timeseries" title="New Haven Line Zip Code Timeseries" width="400"/>
</div>

### NOTE

Note that this does not include zip codes in Connecticut (which the New Haven line covers) since the client is only interested in towns in New York state. 

### Harlem Line

<div style="background-color:white">
<img src="/assets/images/timeseries/harlem_mapTime.png" alt="Harlem Line Zip Code Timeseries" title="Harlem Line Zip Code Timeseries" width="400"/>
</div>

### Hudson Line

<div style="background-color:white">
<img src="/assets/images/timeseries/hudson_mapTime.png" alt="Hudson Line Zip Code Timeseries" title="Hudson Line Zip Code Timeseries" width="400"/>
</div>


## TOP FIVE ZIP CODES

The top five I selected based on the above criteria were 10549, 10573, 10604, 10605, 10706:

## TOP FIVE RECOMMENDATIONS

The top five results that fit the required criteria were 10549, 10573, 10604, 10605, 10706:

## 10549

<div style="background-color:white">
<img src="/assets/images/timeseries/10549.png" alt="timeseries 10549" title="timeseries 10549" width="400"/>
</div>

## 10573

<div style="background-color:white">
<img src="/assets/images/timeseries/10573.png" alt="timeseries 10573" title="timeseries 10573" width="400"/>
</div>

## 10604

<div style="background-color:white">
<img src="/assets/images/timeseries/10604.png" alt="timeseries 10604" title="timeseries 10604" width="400"/>
</div>

## 10605

<div style="background-color:white">
<img src="/assets/images/timeseries/10605.png" alt="timeseries 10605" title="timeseries 10605" width="400"/>
</div>

## 10706

<div style="background-color:white">
<img src="/assets/images/timeseries/10706.png" alt="timeseries 10706" title="timeseries 10706" width="400"/>
</div>

## Top Five Zip Codes in Westchester County

<div style="background-color:white">
<img src="/assets/images/timeseries/top5_final_mapTime.png" alt="top five zipcodes timeseries" title="top five zip codes timeseries" width="400"/>
</div>

## FUTURE WORK

My client was keen on accounting for public school districts, which upon initial inspection would have required a great deal of manual plug and play. However, if there is an API or some other way to scrape this data from the web, I would definitely incorporate school districts as an exogenous factor for the sake of making recommendations for a client. Someone might actually *not* prefer schools with a rating of 10 as these tend to be predominantly all-white. My client in particular was looking for decent school districts below the 10-mark because she wants her child to grow up in a more ethnically-diverse community. Being able to account for such preferences would be part of the future work of this project.

## Contact

You can reach me at <rukeine@gmail.com>.

## License

This project uses the following license: [MIT License](/LICENSE.md).
