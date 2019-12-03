import numpy as np
import pytest
import xarray as xr

import scipy.interpolate
from dcpy.interpolate import pchip, pchip_fillna, pchip_roots


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


def expected_pchip_interpolate_z(data, ix):

    data = data.compute()
    stacked = data.stack({"stacked": ["x", "y"]})

    expected = (
        stacked.isel(z=0).drop_vars("z").expand_dims(z=np.array(ix, ndmin=1)).copy()
    )
    for ii in np.arange(stacked.sizes["stacked"]):
        subset = stacked.isel(stacked=ii)
        mask = np.isnan(subset)
        interpolator = scipy.interpolate.PchipInterpolator(
            x=subset.z.values[~mask], y=subset.values[~mask], extrapolate=False
        )
        expected[:, ii] = interpolator(ix)

    expected = (
        expected.isel(z=slice(len(ix)))
        .assign_coords(z=ix)
        .unstack("stacked")
        .transpose("x", "y", "z")
    )
    return expected


@pytest.mark.parametrize("maybe_chunk", [False, True])
def test_pchip_fillna(data, maybe_chunk):

    data[:, :, 10:20] = np.nan

    if maybe_chunk:
        data = data.chunk({"x": 1, "y": 2})

    actual = pchip_fillna(data, "z")
    expected = expected_pchip_interpolate_z(data, data["z"])
    xr.testing.assert_equal(expected, actual)


@pytest.mark.parametrize(
    "newz",
    [
        -31.23,
        [-31.23],
        np.linspace(-60, 0, 150),
        xr.DataArray(np.linspace(-60, 0, 150), dims="z", name="newz"),
    ],
)
@pytest.mark.parametrize("maybe_chunk", [False, True])
def test_pchip_interpolate(data, maybe_chunk, newz):

    if maybe_chunk:
        data = data.chunk({"x": 1, "y": 2})

    actual = pchip(data, "z", newz)
    expected = expected_pchip_interpolate_z(data, np.array(newz, ndmin=1))
    if isinstance(newz, xr.DataArray):
        assert "newz" in actual.dims
        xr.testing.assert_equal(expected, actual.rename({"newz": "z"}))
    else:
        xr.testing.assert_equal(expected, actual)


# TODO: Multiple roots; check warning
# TODO: weird numpy size > 0 warning
