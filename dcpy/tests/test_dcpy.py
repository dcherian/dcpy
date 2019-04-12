import xarray as xr
import dcpy.ts
import dcpy.util
from ..oceans import calc_wind_input
import numpy as np


def test_xfilter():

    time = np.arange(500)
    da = xr.DataArray(np.random.randn(500,), dims=['time'], coords=[time])

    # test rolling mean
    xr.testing.assert_equal(
        da.rolling(time=20, center=True, min_periods=1).mean(),
        dcpy.ts.xfilter(da, 20, kind='mean', dim='time')
    )

    # test hanning filter
    np.testing.assert_equal(
        dcpy.util.smooth(da.values, 20, 'hanning'),
        dcpy.ts.xfilter(da, 20, kind='hann', dim='time').values
    )

    np.testing.assert_equal(
        dcpy.ts.BandPassButter(da.values, [1/20.0, 1/50.0]),
        dcpy.ts.xfilter(da, [20.0, 50.0], kind='bandpass', dim='time').values
    )


def test_slab_model():
    from ..oceans import coriolis

    ρ0 = 1025
    τx = 1
    τy = 0
    H = 15
    f0 = coriolis(15)
    r = 0.15 * f0
    σ = np.linspace(-4*f0, 4*f0, 200)
    That = np.exp(-(σ/-f0)**2)

    T = (τx + 1j * τy) / ρ0
    Z = (-r + 1j * f0 + σ)/(σ**2 - f0**2  - r**2 - 2*1j*r*σ) * That/H

    _, ZI = calc_wind_input(T, H, f0)
