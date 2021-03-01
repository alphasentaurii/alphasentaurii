---
layout: post
title:  "Starskøpe 4: Fourier Spectrographs"
date:   2020-08-26 11:11:11 -1800
categories: datascience
---


```python
# Estimate period
from astropy import units as u
from astropy.timeseries import BoxLeastSquares
periodogram = BoxLeastSquares.from_timeseries(ts, 'flux')
```

```python
results = periodogram.autopower(0.02 * u.day)
best = np.argmax(results.power)
period = results.period[best]
period
```