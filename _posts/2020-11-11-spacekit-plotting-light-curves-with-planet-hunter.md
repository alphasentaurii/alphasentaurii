---
layout: post
title:  "SPACEKIT Analyzer: plotting light curves"
date:   2020-11-11 11:11:11 -1111
categories: datascience
tags: astrophysics, spacekit
---

## spacekit.Analyzer()
timeseries flux signal analysis

- atomic_vector_plotter: Plots scatter and line plots of time series signal values.
- make_specgram: generate and save spectographs of flux signal frequencies
- planet_hunter: calculate period, plot folded lightcurve from .fits files


### atomic_vector_plotter
Plots scatter and line plots of time series signal values.

```python
from astropy.timeseries import TimeSeries
fits_file = 'ktwo246384368-c12_llc.fits'
ts = TimeSeries.read(fits_file, format='kepler.fits')
flux = ts['sap_flux']
```

Now that we have the flux signal values we can plot them as timeseries scatter and line plots.

```python
from spacekit import analyzer
analyzer = Analyzer()
analyzer.atomic_vector_plotter(flux)

```

<div style="background-color:white">
<img src="/assets/images/spacekit/atomic-vector-scatter.png" alt="atomic vector plotter scatterplot" title="Scatterplot with Planet 1" width="400"/>
</div>

<div style="background-color:white">
<img src="/assets/images/spacekit/atomic-vector-line.png" alt="atomic vector plotter lineplot" title="Lineplot with Planet 1" width="400"/>
</div>

### planet_hunter
calculates period and plots folded light curve from single or multiple .fits files

```python
signal = 'ktwo246384368-c12_llc.fits'
analyzer.planet_hunter(signal, fmt='kepler.fits')
```

<div style="background-color:white">
<img src="/assets/images/spacekit/k2-folded-light-curve.png" alt="k2 folded light curve" title="Planet Hunter K2 Folded Light Curve" width="400"/>
</div>