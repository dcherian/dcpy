# My personal analysis tools

## Highlights

1. Has utility functions for reading various oceanographic datasets as xarray Datasets: see `oceans.py`.
2. Includes dask-aware port of the `seawater` library: see `eos.py`
3. Easy to use spectral plotting with easy switching between segment averaging, frequency band averaging and multitaper spectra: see `ts.py`
4. Some nice plotting functions (`oceans.TSplot`) and plotting utilities: see `plots.py`

## Warning

1. Everything is untested. Not all functions are dask-friendly. PRs welcome!
2. Almost no documentation :(.
3. Some are unpythonic ports of MATLAB functions I used but don't use anymore.
4. Utility reading functions in `oceans.py` have hardcoded directories.
