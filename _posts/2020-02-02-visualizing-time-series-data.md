---
layout: post
title:  "Visualizing Time Series Data"
date:   2020-02-02 02:02:02 -0800
categories: datascience
tags: timeseries project forecasting statistics python
author: Ru Keïn
---

`Time Series Forecasting with SARIMAX and Gridsearch` is a `housing market prediction model` that uses `seasonal ARIMA time-series analysis and GridsearchCV` to `recommend the top 5 zip codes` for purchasing a single-family home in Westchester, New York. The top 5 zip code recommendations rely on the following factors: highest ROI, lowest confidence intervals, and shortest commute time from Grand Central Station. Along with several custom time series analysis helper functions I wrote for this project, I also extrapolate the USZIPCODE pypi library to account for several exogenous factors, including average income levels. 

## Making Forecast Predictions by Zip Code

<div style="background-color:white">
<img src="/assets/images/timeseries/10605.png" alt="forecast predictions by zip code" title="Forecast Predictions by Zip Code" width="400"/>
</div>

## Summary

Prior to training, I set out to identify trends, seasonality, and autoregression elements within the Zillow Housing Market Dataset. I then use a fitting procedure to find the coefficients of a regression model, including several plots and statistical tests of the residual errors along the way. 

## Prerequisites

Before you begin, ensure you have met the following requirements:
* You have installed the latest version of `Jupyter Notebook`
* You have a `<Windows/Linux/Mac>` machine. 


## Running the Time Series Forecasting with SARIMAX and Gridsearch Project

To run this project locally, follow these steps:

* In the command line/terminal:

```
$ git clone https://github.com/alphasentaurii/timeseries-forecasting-with-sarimax-and-gridsearch
$ cd timeseries-forecasting-with-sarimax-and-gridsearch
$ jupyter notebook
```

Please note that the Zillow data set contains *millions* of US Zipcodes. If you want to fork this project and follow along with some of the steps I took, you can apply the same model I did to a completely different county. The only part that you'll need to skip (or adjust) is the Metro North railroad section, since this only applies to Westchester County, New York. 

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
2. Lowest Confidence Intervals (using the prediction mean)
3. Risk mitigation: financial stability of homeowners and homes based on historic data

## Objective

Make forecast predictions for zip codes and their mean home values using Seasonal ARIMA time-series analysis.

## Commute Times

Since commute time to Grand Central Station is part of the client's required criteria, I first had to look up which towns/zip codes were on which train lines. Grand Central has 3 main lines on Metro North Railroad: Hudson, Harlem, and New Haven. The first question I was interested in answering was if the average home prices for the zip codes that fall under these geographic sections display any trends.

## Model

Make forecast predictions for zip codes and their mean home values using Seasonal ARIMA time-series analysis.

### 1. Model Identification

* Plots and summary statistics
* Identify trends, seasonality, and autoregression elements
* Get an idea of the amount of differencing and size of lag that will be required.

### 2. Parameter Estimation

* Use a fitting procedure to find the coefficients of the regression model.
* split data into train and test sets.

### 3. Model Checking

* Plots and statistical tests of the residual errors
* Determine the amount and type of temporal structure not captured by the model.

### 4. Forecasting

* Input complete time-series and get prediction values
* Identify top 5 zip codes based on required criteria (above).

## Workflow

#### IMPORT
- Import libraries, functions and dataset

#### PREPROCESSING
- Reshape data format (Wide to Long): `pd.melt()` function
- convert datetime column to datetime objects
- set datetime col to index

#### RESAMPLING
- Groupby state, county, cities (`get_groups()`)
- filter by zipcodes in selected New York cities in Westchester County
- EDA & visualizations

#### PARAMETERS
- seasonal decomposition
- decomposed residuals
- plot time series
- Check for stationarity
- pmdarima (differencing)

#### INITIAL MODEL
- Split train/test by integer index based on len(data)//test_size
- Train ARIMA model
- Model results/summary for each independent zipcode

#### REVISED MODEL
- Seasonal Arima (SARIMA)
- Exogenous Data
- SARIMAX

#### FORECAST
- Get predictions for test set.
- Get another set of predictions built off of train+test set combined.

#### INTERPRET
- Analyze Results
- Summarize Findings
- Make Recommendations

## Mean Values by Train Line

Since commute time to Grand Central Station is part of the client's required criteria, I first had to look up which towns/zip codes were on which train lines. Grand Central has 3 main lines on Metro North Railroad: Hudson, Harlem, and New Haven. The first question I was interested in answering was if the average home prices for the zip codes that fall under these geographic sections display any trends. 

### Area Plot

<div style="background-color:white">
<img src="/assets/images/timeseries/meanvalues_area.png" alt="Mean Values by Train Line Area Plot" title="Mean Values by Train Line Area Plot" width="400"/>
</div>

### New Haven line 

<div style="background-color:white">
<img src="/assets/images/timeseries/newhaven_mapTime.png" alt="New Haven Line Zip Code Timeseries" title="New Haven Line Zip Code Timeseries" width="400"/>
</div>

#### NOTE

Note that the plot above _does not_ include zip codes in Connecticut (which the New Haven line covers) since the client is only interested in towns in New York state.

### Harlem Line

<div style="background-color:white">
<img src="/assets/images/timeseries/harlem_mapTime.png" alt="Harlem Line Zip Code Timeseries" title="Harlem Line Zip Code Timeseries" width="400"/>
</div>

### Hudson Line

<div style="background-color:white">
<img src="/assets/images/timeseries/hudson_mapTime.png" alt="Hudson Line Zip Code Timeseries" title="Hudson Line Zip Code Timeseries" width="400"/>
</div>

## Custom Time Series Analysis Functions (python)

Below is the `mapTime()` function I wrote for generating the timeseries plots above.

#### `mapTime()`

```python
def mapTime(d, xcol, ycol='MeanValue', X=None, vlines=None, MEAN=True):
    """
    'Maps' a timeseries plot of zipcodes, you can input either a dataframe
    or a dictionary of dataframes.

    
    # fig,ax = mapTime(d=HUDSON, xcol='RegionName', ycol='MeanValue', 
                       MEAN=True, vlines=None)
    
    **ARGS
    d: takes a dictionary of dataframes OR a single dataframe
    xcol: column in dataframe containing x-axis values (ex: zipcode)
    ycol: column in dataframe containing y-axis values (ex: price)
    X: list of x values to plot on x-axis (defaults to all x in d if empty)
    
    **kw_args
    mean: plots the mean of X (default=True)
    vlines : default is None: shows MIN_, MAX_, crash 
    
    *Ex1: `d` = dataframe
    mapTime(d=NY, xcol='RegionName', ycol='MeanValue', X=list_of_zips)
    
    *Ex2: `d` = dictionary of dataframes
    mapTime(d=NYC, xcol='RegionName', y='MeanValue')
    """
  
    
    # create figure for timeseries plot
    fig, ax = plt.subplots(figsize=(21,13))
    plt.title(label=f'Time Series Plot: {str(ycol)}')
    ax.set(title='Mean Home Values', xlabel='Year', ylabel='Price($)', font_dict=font_title)  
    
    zipcodes = []
    #check if `d` is dataframe or dictionary
    if type(d) == pd.core.frame.DataFrame:
        # if X is empty, create list of all zipcodes
        if len(X) == 0:
            zipcodes = list(d[xcol].unique())
        else:
            zipcodes = X
        # cut list in half  
        breakpoint = len(zipcodes)//2
        
        for zc in zipcodes:
            if zc < breakpoint:
                ls='-'
            else:
                ls='--'
            ts = d[zc][ycol].rename(zc)#.loc[zc]
            ts = d[ycol].loc[zc]
            ### PLOT each zipcode as timeseries `ts`
            ts.plot(label=str(zc), ax=ax, ls=ls)
        ## Calculate and plot the MEAN
        
        if MEAN:
            mean = d[ycol].mean(axis=1)
            mean.plot(label='Mean',lw=5,color='black')
    
    elif type(d) == dict:
        # if X passed in as empty list, create list of all zipcodes
        if len(X) == 0:
            zipcodes = list(d.keys())
        else:
            zipcodes = X
        # cut list in half  
        breakpoint = len(zipcodes)//2
        
        # create empty dictionary for plotting 
        txd = {}
        # create different linestyles for zipcodes (easier to distinguish if list is long)
        for i,zc in enumerate(zipcodes):
            if i < breakpoint:
                ls='-'
            else:
                ls='--'
            # store each zipcode as ts  
            ts = d[zc][ycol].rename(zc)
            ### PLOT each zipcode as timeseries `ts`
            ts.plot(label=str(zc), ax=ax, ls=ls, lw=2);
            txd[zc] = ts
            
        if MEAN:
            mean = pd.DataFrame(txd).mean(axis=1)
            mean.plot(label='Mean',lw=5,color='black')
            
    ax.legend(bbox_to_anchor=(1.04,1), loc="upper left", ncol=2)
            
    if vlines:
        ## plot crash, min and max vlines
        crash = '01-2009'
        ax.axvline(crash, label='Housing Index Drops',color='red',ls=':',lw=2)
        MIN_ = ts.loc[crash:].idxmin()
        MAX_ = ts.loc['2004':'2010'].idxmax()
        ax.axvline(MIN_, label=f'Min Price Post Crash {MIN_}', color='black',lw=2)    
        ax.axvline(MAX_,label='Max Price', color='black', ls=':',lw=2) 

    return fig, ax
```

## Seasonality and Trends

To check for seasonality and remove trends, I also wrote a custom function to generate all the necessary time series analysis plots in one shot. This is really handy for not having to repeat steps over and over again (um, yes that is the point of any function...):

### Autocorrelation

<div style="background-color:white">
<img src="/assets/images/timeseries/output_50_1.png" alt="autocorrelation" title="Autocorrelation" width="400"/>
</div>

### Differencing

<div style="background-color:white">
<img src="/assets/images/timeseries/output_50_2.png" alt="differencing" title="differencing" width="400"/>
</div>

### Seasonality

<div style="background-color:white">
<img src="/assets/images/timeseries/output_50_3.png" alt="seasonality" title="seasonality" width="400"/>
</div>

### Partial Autocorrelation

<div style="background-color:white">
<img src="/assets/images/timeseries/output_50_4.png" alt="partial autocorrelation" title="Partial Autocorrelation" width="400"/>
</div>

### Rolling Mean and Standard Deviation

<div style="background-color:white">
<img src="/assets/images/timeseries/output_50_5.png" alt="rolling mean and standard deviation" title="Rolling Mean and Standard Deviation" width="400"/>
</div>

### Check for Seasonality

<div style="background-color:white">
<img src="/assets/images/timeseries/output_50_6.png" alt="check for seasonality" title="Checking for Seasonality" width="400"/>
</div>

### Residual, Density, QQ, and Correlogram

<div style="background-color:white">
<img src="/assets/images/timeseries/output_65_0.png" alt="residual density qq and correlogram" title="Residual, Density, QQ, and Correlogram" width="400"/>
</div>

## Creating the Visuals with `clockTime()`

All of the above are generated in one shot with the clockTime() function:

```python
def clockTime(ts, lags, d, TS, y):
    """    
     /\    /\    /\    /\  ______________/\/\/\__-_-_
    / CLOCKTIME STATS /  \/
        \/    \/    \/    

    # clockTime(ts, lags=43, d=5, TS=NY, y='MeanValue',figsize=(13,11))
    #
    # ts = df.loc[df['RegionName']== zc]["MeanValue"].rename(zc).resample('MS').asfreq()
    """
    # import required libraries
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    from numpy import log 
    import pandas as pd
    from pandas import Series
    from pandas.plotting import autocorrelation_plot
    from pandas.plotting import lag_plot
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf
    from statsmodels.graphics.tsaplots import plot_pacf  
    
    print(' /\\   '*3+' /')
    print('/ CLOCKTIME STATS')
    print('    \/'*3)

    #**************#   
    # Plot Time Series
    #original
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(21,13))
    ts.plot(label='Original', ax=axes[0,0],c='red')
    # autocorrelation 
    autocorrelation_plot(ts, ax=axes[0,1], c='magenta') 
    # 1-lag
    autocorrelation_plot(ts.diff().dropna(), ax=axes[1,0], c='green')
    lag_plot(ts, lag=1, ax=axes[1,1])
    
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show();
    
    # DICKEY-FULLER Stationarity Test
    # TS = NY | y = 'MeanValue'
    dtest = adfuller(TS[y].dropna())
    if dtest[1] < 0.05:
        ## difference data before checking autoplot
        stationary = False
        r = 'rejected'
    else:
        ### skip differencing and check autoplot
        stationary = True 
        r = 'accepted'

    #**************#
    # ts orders of difference
    ts1 = ts.diff().dropna()
    ts2 = ts.diff().diff().dropna()
    ts3 = ts.diff().diff().diff().dropna()
    ts4 = ts.diff().diff().diff().diff().dropna()
    tdiff = [ts1,ts2,ts3,ts4]
    # Calculate Standard Deviation of Differenced Data
    sd = []
    for td in tdiff:
        sd.append(np.std(td))
    
    #sd = [np.std(ts1), np.std(ts2),np.std(ts3),np.std(ts4)]
    SD = pd.DataFrame(data=sd,index=['ts1',' ts2', 'ts3', 'ts4'], columns={'sd'})
    #SD['sd'] = [np.std(ts1), np.std(ts2),np.std(ts3),np.std(ts4)]
    SD['D'] = ['d=1','d=2','d=3','d=4']
    MIN = SD.loc[SD['sd'] == np.min(sd)]['sd']

    # Extract and display full test results 
    output = dict(zip(['ADF Stat','p-val','# Lags','# Obs'], dtest[:4]))
    for key, value in dtest[4].items():
        output['Crit. Val (%s)'%key] = value
    output['min std dev'] = MIN
    output['NULL HYPOTHESIS'] = r
    output['STATIONARY'] = stationary
     
    # Finding optimal value for order of differencing
    from pmdarima.arima.utils import ndiffs
    adf = ndiffs(x=ts, test='adf')
    kpss = ndiffs(x=ts, test='kpss')
    pp = ndiffs(x=ts, test='pp')
        
    output['adf,kpss,pp'] = [adf,kpss,pp]

    #**************#
    # show differencing up to `d` on single plot (default = 5)
    fig2 = plt.figure(figsize=(13,5))
    ax = fig2.gca()
    for i in range(d):
        ax = ts.diff(i).plot(label=i)
    ax.legend(bbox_to_anchor=(1.04,1), loc="upper left", ncol=2)
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show();
    
    #**************#
    # DIFFERENCED SERIES
    fig3 = plt.figure(figsize=(13,5))
    ts1.plot(label='d=1',figsize=(13,5), c='blue',lw=1,alpha=.7)
    ts2.plot(label='d=2',figsize=(13,5), c='red',lw=1.2,alpha=.8)
    ts3.plot(label='d=3',figsize=(13,5), c='magenta',lw=1,alpha=.7)
    ts4.plot(label='d=4',figsize=(13,5), c='green',lw=1,alpha=.7)
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=True, 
               fancybox=True, facecolor='lightgray')
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show();
    #**************#
    
    # Plot ACF, PACF
    fig4,axes = plt.subplots(nrows=2, ncols=2, figsize=(21,13))
    plot_acf(ts1,ax=axes[0,0],lags=lags)
    plot_pacf(ts1, ax=axes[0,1],lags=lags)
    plot_acf(ts2,ax=axes[1,0],lags=lags)
    plot_pacf(ts2, ax=axes[1,1],lags=lags)
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show();
    
    #**************#
    # plot rolling mean and std
    #Determine rolling statistics
    rolmean = ts.rolling(window=12, center=False).mean()
    rolstd = ts.rolling(window=12, center=False).std()
        
    #Plot rolling statistics
    fig = plt.figure(figsize=(13,5))
    orig = plt.plot(ts, color='red', label='original')
    mean = plt.plot(rolmean, color='cyan', label='rolling mean')
    std = plt.plot(rolstd, color='orange', label='rolling std')
    
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left") 
    plt.title('Rolling mean and standard deviation')
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show();
    
    #**************#
    # # Check Seasonality 
    """
    Calculates and plots Seasonal Decomposition for a time series
    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    decomp = seasonal_decompose(ts, model='additive') # model='multiplicative'

    decomp.plot()
    ts_seas = decomp.seasonal

    ax = ts_seas.plot(c='green')
    fig = ax.get_figure()
    fig.set_size_inches(13,11)

    ## Get min and max idx
    min_ = ts_seas.idxmin()
    max_ = ts_seas.idxmax()
    min_2 = ts_seas.loc[max_:].idxmin()

    ax.axvline(min_,label=min_,c='red')
    ax.axvline(max_,c='red',ls=':', lw=2)
    ax.axvline(min_2,c='red', lw=2)

    period = min_2 - min_
    ax.set_title(f'Season Length = {period}')

    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show();
   
    #*******#
    clock = pd.DataFrame.from_dict(output, orient='index')
    print(' /\\   '*3+' /')
    print('/ CLOCK-TIME STATS')
    print('    \/'*3)
    
    #display results
    print('---'*9)
    return clock

```


## GridsearchCV with SARIMAX

I then ran a gridsearch using a Seasonal ARIMA (SARIMAX) model to make forecast predictions on all 61 zip codes in Westchester County. Using PANDAS, I narrowed down the list by top 10 highest ROI zip codes. I then identified which of these had the lowest confidence intervals in order to ensure I was only selecting the most accurate results.

### Confidence Intervals and ROI Predictions (3D Graph)

<div style="background-color:white">
<img src="/assets/images/timeseries/conf_roi_pred_3D.png" alt="conf roi pred 3D" title="Confidence Intervals and ROI Predictions in 3D" width="400"/>
</div>

### Confidence Intervals and ROI (Heatmap)

<div style="background-color:white">
<img src="/assets/images/timeseries/conf_roi_heatmap.png" alt="conf roi pred heatmap" title="Confidence Intervals and ROI Predictions Heatmap" width="400"/>
</div>

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

If you want to contact me you can reach me at <rukeine@gmail.com>.

## License

This project uses the [MIT License](/LICENSE.md).

```python
                       
           /\    _       _                           _                      *  
/\_/\_____/  \__| |_____| |_________________________| |___________________*___
[===]    / /\ \ | |  _  |  _  | _  \/ __/ -__|  \| \_  _/ _  \ \_/ | * _/| | |
 \./    /_/  \_\|_|  ___|_| |_|__/\_\ \ \____|_|\__| \__/__/\_\___/|_|\_\|_|_|
                  | /             |___/        
                  |/   
```