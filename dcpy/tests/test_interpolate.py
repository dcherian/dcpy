import numpy as np
import pytest
import scipy.interpolate
from dcpy.interpolate import bin_to_new_coord, pchip, pchip_fillna, pchip_roots

import xarray as xr


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

    assert ix.ndim == 1
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
        xr.DataArray(np.linspace(-60, 0, 150), dims="target", name="newz"),
        (
            xr.DataArray(
                np.linspace(-60, 0, 150), dims="target", name="newz"
            ).expand_dims(x=2, y=3)
        ),
    ],
)
@pytest.mark.parametrize("maybe_chunk", [False, True])
def test_pchip_interpolate(data, maybe_chunk, newz):

    if isinstance(newz, xr.DataArray) and "x" in newz.dims:
        ix_nd_da = True
    else:
        ix_nd_da = False

    if ix_nd_da:
        # hack for pytest.parametrize & xarray align
        # assign coords so alignment works
        newz["x"] = data.x
        newz["y"] = data.y

    if maybe_chunk:
        data = data.chunk({"x": 1, "y": 2})
        if ix_nd_da:
            # TODO : is this necessary?
            newz = newz.chunk({"x": 1, "y": 2})

    actual = pchip(data, "z", newz)

    if ix_nd_da:
        # expected_pchip_interpolate_z expects 1D values.
        # so I subset now and broadcast later before comparing
        # could be avoided by modifying expected_pchip_interpolate_na
        expected_newz = newz.isel(x=0, y=0).drop(["x", "y"]).values
    else:
        expected_newz = np.array(newz, ndmin=1)

    expected = expected_pchip_interpolate_z(data, expected_newz)

    if ix_nd_da:
        expected = expected.rename({"z": "target"})
        expected["z_target"] = newz
        expected = expected.drop("target")

    if isinstance(newz, xr.DataArray):
        assert "target" in actual.dims
        if "target" not in expected.dims:
            expected = expected.rename({"z": "target"})
        xr.testing.assert_equal(expected, actual)
    else:
        xr.testing.assert_equal(expected, actual)


# TODO: test errors
# TODO: test unify_chunks
# TODO: Multiple roots; check warning
# TODO: weird numpy size > 0 warning
# TODO: test with datasets and coords with the interpolated dimension


@pytest.mark.xfail
def test_bin_to_new_coord():
    depth = xr.DataArray(np.arange(-20, 0, 1), dims=["depth"])
    z0 = xr.DataArray(np.random.randint(-20, 0, (10,)) * 1.0, dims=("time",))
    data = depth - z0
    new_coord = depth - z0
    new_coord[1, 4] = np.nan
    new_coord[4, 5] = np.nan
    data["zeuc"] = new_coord

    actual = bin_to_new_coord(
        data, old_coord="depth", new_coord="zeuc", edges=np.arange(-4.5, 4.6, 1)
    )

    expected = xr.ones_like(z0) * new_1d_coord
    expected = expected.where(
        (expected >= data.min("depth")) & (expected <= data.max("depth"))
    )
    expected[new_dim] = new_1d_coord
    # stick expected NaNs in the right place. This really should be better.
    if data[1, 4].values >= edges.min() and data[1, 4].values <= edges.max():
        expected.loc[{"zeuc": data[1, 4], "time": 4}] = np.nan
    if data[4, 5].values >= edges.min() and data[4, 5].values <= edges.max():
        expected.loc[{"zeuc": data[4, 5], "time": 5}] = np.nan

    xr.testing.assert_equal(expected, actual)
