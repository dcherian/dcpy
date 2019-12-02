import numpy as np
import pytest
import xarray as xr

import scipy.interpolate
from dcpy.interpolate import pchip_fillna, pchip_roots


@pytest.fixture
def data():
    x = xr.DataArray(np.linspace(-10, 10, 2), dims="x")
    y = xr.DataArray(np.linspace(-1, 15, 3), dims="y")
    z = xr.DataArray(np.linspace(-60, 0, 60), dims="z")

    z0 = -31.234
    data = (z - z0).expand_dims(x=x, y=y).assign_coords(z=z, z0=z0).copy()
    return data


@pytest.mark.parametrize("maybe_chunk", [False, True])
@pytest.mark.parametrize("targets", [0, [0.0], [-30], [-30, 29.234]])
def test_roots(data, targets, maybe_chunk):

    if maybe_chunk:
        data = data.chunk({"x": 1, "y": 2})

    actual = pchip_roots(data, dim="z", target=targets)
    coord = xr.DataArray(np.array(targets, ndmin=1), dims="target")
    coord = coord.assign_coords(target=coord)

    expected = (coord + data.z0).broadcast_like(data.isel(z=0))
    expected = expected.where(expected > data.z.min())

    xr.testing.assert_equal(expected, actual)


@pytest.mark.parametrize("maybe_chunk", [False, True])
def test_pchip_fillna(data, maybe_chunk):

    data[:, :, 10:20] = np.nan

    if maybe_chunk:
        data = data.chunk({"x": 1, "y": 2})

    actual = pchip_fillna(data, "z")

    data = data.compute()
    stacked = data.stack({"stacked": ["x", "y"]})
    expected = stacked.copy(deep=True)
    for ii in np.arange(stacked.sizes["stacked"]):
        subset = stacked.isel(stacked=ii)
        mask = np.isnan(subset)
        interpolator = scipy.interpolate.PchipInterpolator(
            x=subset.z.values[~mask], y=subset.values[~mask], extrapolate=False
        )
        expected[:, ii] = interpolator(subset.z)

    xr.testing.assert_equal(
        expected.unstack("stacked").transpose("x", "y", "z"), actual
    )


# TODO: Multiple roots; check warning
# TODO: weird numpy size > 0 warning
