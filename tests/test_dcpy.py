import xarray as xr
import dcpy.ts
import dcpy.util
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
