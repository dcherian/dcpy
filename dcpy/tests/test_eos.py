import dask
import numpy as np
import pytest
import seawater as sw
import xarray as xr

from dcpy import eos


@pytest.mark.parametrize(
    "S",
    [
        35,
        np.array([35.0, 35.1, 32.2]),
        xr.DataArray([35.0, 35.1, 32.2]),
        xr.DataArray([35.0, 35.1, 32.2]).chunk({"dim_0": 1}),
    ],
)
@pytest.mark.parametrize(
    "T",
    [
        20,
        np.array([30.0, 20.1, 25.2]),
        xr.DataArray([30.0, 20.1, 25.2]),
        xr.DataArray([35.0, 35.1, 32.2]).chunk({"dim_0": 1}),
    ],
)
@pytest.mark.parametrize(
    "P",
    [
        0,
        np.array([0, 100, 200]),
        xr.DataArray([0.0, 100.0, 200.0]),
        xr.DataArray([35.0, 35.1, 32.2]).chunk({"dim_0": 1}),
    ],
)
@pytest.mark.parametrize("pt", [True, False])
@pytest.mark.parametrize("func", ["alpha", "beta"])
def test_eos(func, S, T, P, pt):
    actual = getattr(eos, func)(S, T, P, pt)
    expected = getattr(sw, func)(S, T, P, pt)

    if np.any(
        [
            isinstance(S, xr.DataArray),
            isinstance(T, xr.DataArray),
            isinstance(P, xr.DataArray),
        ]
    ):
        assert isinstance(actual, xr.DataArray)

        if np.any(
            [
                isinstance(S, dask.array.Array),
                isinstance(T, dask.array.Array),
                isinstance(P, dask.array.Array),
            ]
        ):
            assert isinstance(actual.data, dask.array)

        np.testing.assert_equal(actual.values, expected)
    else:
        np.testing.assert_equal(actual, expected)
