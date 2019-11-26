# Note that guvectorized functions must be defined in a module for dask distributed :/
# see https://github.com/numba/numba/issues/4314

# slightly modified from xyzpy

from xarray import apply_ufunc
from numba import njit, guvectorize, double, int_, jitclass
import numpy as np
from scipy import interpolate


@njit
def preprocess_nan_func(x, y, out):  # pragma: no cover
    """Pre-process data for a 1d function that doesn't accept nan-values.
    """
    # strip out nan
    mask = np.isfinite(x) & np.isfinite(y)
    num_nan = np.sum(~mask)

    if x.size - num_nan < 2:
        out[:] = np.nan
        return
    elif num_nan != 0:
        x = x[mask]
        y = y[mask]

    return x, y


@guvectorize(
    [
        (int_[:], int_[:], double[:], double[:]),
        (double[:], int_[:], double[:], double[:]),
        (int_[:], double[:], double[:], double[:]),
        (double[:], double[:], double[:], double[:]),
    ],
    "(n),(n),(m)->(m)",
    forceobj=True,
)
def _gufunc_pchip_roots(x, y, target, out=None):  # pragma: no cover
    xy = preprocess_nan_func(x, y, out)
    if xy is None:
        return
    x, y = xy

    # interpolating function
    ifn = interpolate.PchipInterpolator(x, y - targets, extrapolate=False)
    out[:] = ifn(ix)


@guvectorize(
    [
        (int_[:], int_[:], double[:], double[:]),
        (double[:], int_[:], double[:], double[:]),
        (int_[:], double[:], double[:], double[:]),
        (double[:], double[:], double[:], double[:]),
    ],
    "(n),(n),(m)->(m)",
    forceobj=True,
)
def _gufunc_pchip(x, y, ix, out=None):  # pragma: no cover
    xy = preprocess_nan_func(x, y, out)
    if xy is None or len(x) < 2:
        return
    x, y = xy

    # interpolating function
    ifn = interpolate.PchipInterpolator(x, y, extrapolate=False)
    out[:] = ifn(ix)


def pchip(obj, dim, ix):
    """
    Interpolate along axis ``dim`` using :func:`scipy.interpolate.pchip`.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        The object to interpolate.
    dim : str
        The dimension to interpolate along.
    ix : DataArray
        If int, interpolate to this many points spaced evenly along the range
        of the original data. If array, interpolate to those points directly.

    Returns
    -------
    new_xobj : xarray.DataArray or xarray.Dataset
    """

    input_core_dims = [(dim,), (dim,), (dim,)]
    args = (obj[dim], obj, ix)

    output_core_dims = [("__temp_dim__",)]

    result = apply_ufunc(
        _gufunc_pchip,
        *args,
        input_core_dims=input_core_dims,
        output_core_dims=output_core_dims,
        dask="parallelized",
        output_dtypes=[float],
        output_sizes={"__temp_dim__": len(ix)},
    )
    result["__temp_dim__"] = ix.values
    return result.rename({"__temp_dim__": dim})


def pchip_fillna(obj, dim):
    return pchip(obj, dim, obj[dim])
