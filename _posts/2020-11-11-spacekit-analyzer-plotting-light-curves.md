---
layout: post
title:  "SPACEKIT Analyzer: plotting light curves"
date:   2020-11-11 11:11:11 -1111
categories: datascience
tags: astrophysics spacekit
author: Ru Keïn
---

# timeseries flux signal analysis

In the [previous post]({% link _posts/2020-10-10-spacekit-radio-scraping-nasa-api.md %}), we downloaded a set of K2 confirmed planet Fits files into a local directory './data/mast'. Now that we have our datasets, we can extract the time-series flux data and use this to generate some plots. Initially, we'll plot some basic scatter and line plots. Next we'll apply time-binned phase folding to identify periodicity and plot the normalized light curves in order to identify potential TCEs (threshold crossing events: when an object orbiting the star causes the light flux values (our data) to dip signicantly on a periodic basis). These events are highly indicative of a planet orbiting the star, making this a standard approach to identifying exoplanets.


First we'll extract the long cadence signals then plot them as line and scatter plots. An exoplanet candidate will usually display regular (periodic) dips, though a long period may only include a single dip.

```python
from spacekit.analyzer.explore import SignalPlots
sp = SignalPlots(show=True)
fits_file = os.path.join(data, 'ktwo206181769-c03_llc.fits')
timestamps, flux = sp.read_ts_signal(fits_file, signal_col="sap_flux")
sp.atomic_vector_plotter(flux, timestamps=timestamps, name="k2_llc_sap")
```


<div style="background-color:white">
<img src="/assets/images/spacekit/c03-206181769-sap_flux.png" alt="K2 timeseries SAP Flux" title="K2 timeseries SAP Flux" width="400"/>
</div>


We can also look at an error-corrected and denoised version of the signal using the PDCSAP_FLUX values:

```python
timestamps, pflux = sp.read_ts_signal(fits_file, signal_col="pdcsap_flux", bkjd=True, remove_nans=True)
sp.atomic_vector_plotter(flux, timestamps=timestamps, name="k2_llc_pdcsap")
```


<div style="background-color:white">
<img src="/assets/images/spacekit/c03-206181769-pdcsap-flux.png" alt="K2 timeseries PDCSAP Flux" title="K2 timeseries PDCSAP Flux" width="400"/>
</div>


If we have a labeled dataset (e.g. first index in each array is a target label indicating 2=planet and 1=no planet), we can include that data in our plots as well:

```python
sp = SignalPlots(show=True, target_cns={1: "No Planet", 2: "Planet"}, color_map={1: "red", 2: "blue"})
# if target class is at the first index of our timeseries array
sp.atomic_vector_plotter(flux[1:], label=flux[0], x_units="Time", y_units="PDC_SAP Flux")
```

<div style="background-color:white">
<img src="/assets/images/spacekit/atomic-vector-line.png" alt="atomic vector plotter lineplot" title="Lineplot with Planet 1" width="400"/>
</div>

<div style="background-color:white">
<img src="/assets/images/spacekit/atomic-vector-scatter.png" alt="atomic vector plotter scatterplot" title="Scatterplot with Planet 1" width="400"/>
</div>


## phase folded light curves

For a more robust analysis, we can calculate the period and plot a time-binned and phase-folded light curve. This can often (though not always) confirm the presence of at least one transiting exoplanet.

```python
flist = ['ktwo206181769-c03_llc.fits']
df = sp.signal_phase_folder(flist)
sp.plot_phase_signals(df.iloc[0])
```

<div style="background-color:white">
<img src="/assets/images/spacekit/k2-folded-light-curve.png" alt="k2 folded light curve" title="Planet Hunter K2 Folded Light Curve" width="400"/>
</div>
