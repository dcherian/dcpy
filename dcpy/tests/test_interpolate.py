import numpy as np
import pytest
import xarray as xr

from ..dcpy.interpolate import pchip_roots


@pytest.fixture
def data():
    x = xr.DataArray(np.linspace(-10, 10, 10), dims="x")
    y = xr.DataArray(np.linspace(-1, 15, 5), dims="y")
    z = xr.DataArray(np.linspace(-60, 0, 60), dims="z")

    z0 = -31.234
    data = (0 * x + 0 * y + (z - z0)).assign_coords(z=z, z0=z0)
    return data


@pytest.mark.parametrize("targets", [0, [0.0], [-30], [0.0, -15, -30, 29.234]])
def test_roots(data, targets):

    actual = pchip_roots(data, dim="z", target=targets)
    expected = (depths.target + data.z0).broadcast_like(data.isel(z=0))
    expected = expected.where(expected < 0)
    xr.testing.assert_equal(expected, actual)
