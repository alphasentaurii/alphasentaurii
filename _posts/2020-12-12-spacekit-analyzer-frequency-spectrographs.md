---
layout: post
title:  "SPACEKIT Analyzer: frequency spectrographs"
date:   2020-12-12 12:12:12 -1111
categories: datascience
tags: spacekit astrophysics
author: Ru Keïn
---

# Plotting a SpecGram

Taking our [previous analysis]({% link _posts/2020-11-11-spacekit-analyzer-plotting-light-curves.md %}) one step further, we might also be interested in transforming our signal from the time domain into the frequency domain. We can take the same data as before to generate and save spectrographs of our flux signal frequencies.

```python
from spacekit.analyzer.explore import SignalPlots
sp = SignalPlots(show=True)
fits_file = os.path.join(data, 'ktwo206181769-c03_llc.fits')
_, flux = sp.read_ts_signal(fits_file, signal_col="sap_flux")
sp.flux_specs(flux, cmap="plasma", num="206181769")
```

<div style="background-color:white">
<img src="/assets/images/spacekit/c03-206181769_specgram.png" alt="k2 light curve frequency specgram" title="K2 llc specgram" width="400"/>
</div>


# SpecGrams for Machine Learning Image Classification

If we want to use the plots for machine learning image classification, we can make them frameless with no axes so the inputs are raw RGB color values.

```python
sp.flux_specs(flux, cmap="plasma", save_for_ML=True,fname="c03-206181769-ml.png")
```

<div style="background-color:transparent">
<img src="/assets/images/spacekit/c03-206181769-ml.png" alt="k2 light curve frequency specgram for ML image classification" title="K2 llc specgram for ML image classification" width="400"/>
</div>

Finally, it's also possible to generate these in grayscale by leaving out the color map kwarg - this can be beneficial for reducing memory usage with ML use cases in particular.

```python
sp.flux_specs(flux, save_for_ML=True,fname="c03-206181769-bw.png")
```

<div style="background-color:transparent">
<img src="/assets/images/spacekit/c03-206181769-bw.png" alt="k2 light curve frequency specgram grayscale" title="K2 llc specgram grayscale" width="400"/>
</div>
