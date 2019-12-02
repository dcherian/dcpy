# These pchip implementations are simplified from those at xyzpy and add dask support
# https://github.com/jcmgray/xyzpy/blob/develop/xyzpy/signal.py

# Note that guvectorized functions must be defined in a module for dask distributed :/
# see https://github.com/numba/numba/issues/4314

# guvectorized functions have out=None, you need to assign to out. no returns allowed!
# slightly modified from xyzpy

import xarray as xr
from numba import njit, guvectorize, double, int_, jitclass
import numpy as np
from scipy import interpolate
import warnings
from typing import Any, Iterable


def is_scalar(value: Any, include_0d: bool = True) -> bool:
    """Whether to treat a value as a scalar.

    Any non-iterable, string, or 0-D array
    """
    if include_0d:
        include_0d = getattr(value, "ndim", None) == 0
    return (
        include_0d
        or isinstance(value, (str, bytes))
        or not (isinstance(value, (Iterable,)) or hasattr(value, "__array_function__"))
    )


@njit
def preprocess_nan_func(x, y, out):  # pragma: no cover
    """Pre-process data for a 1d function that doesn't accept nan-values.
    """
    # strip out nan
    mask = np.isfinite(x) & np.isfinite(y)
    num_nan = np.sum(~mask)

    if x.size - num_nan < 2:
        return None
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
def _gufunc_pchip_roots(x, y, target, out):  # pragma: no cover
    xy = preprocess_nan_func(x, y, out)
    if xy is None:
        out[:] = np.nan
        return
    x, y = xy

    # reshape to [target, ...]
    target = np.reshape(target, [len(target)] + [1,] * y.ndim)
    y = y[np.newaxis, ...]

    interpolator = interpolate.PchipInterpolator(
        x, y - target, extrapolate=False, axis=-1,
    )
    roots = interpolator.roots()
    flattened = roots.ravel()
    for idx, f in enumerate(flattened):
        if f.size > 1:
            warnings.warn(
                "Found multiple roots. Picking the first one. This will depend on the ordering of `dim`",
                UserWarning,
            )
            flattened[idx] = f[0]
    good = flattened.nonzero()[0]
    out[:] = np.where(
        np.isin(np.arange(flattened.size), good), flattened, np.nan
    ).reshape(roots.shape)


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
    min_points = 2  # TODO: make this a kwarg
    if xy is None or len(x) < min_points:
        out[:] = np.nan
        return
    x, y = xy

    interpolator = interpolate.PchipInterpolator(x, y, extrapolate=False)
    out[:] = interpolator(ix)


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

    result = xr.apply_ufunc(
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


def pchip_roots(obj, dim, target):
    """
    Find locations where `obj == target` along dimension `dim`.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        The object to interpolate.
    dim : str
        The axis to interpolate along.
    target : target values to locate
        Locates values by constructing PchipInterpolant and solving for roots.

    Returns
    -------
    new_xobj : xarray.DataArray or xarray.Dataset
    """

    if is_scalar(target):
        target = np.array(target, ndmin=1)

    if isinstance(target, (np.ndarray, list)):
        target = xr.DataArray(target, dims="target")

    assert target.ndim == 1

    input_core_dims = [(dim,), (dim,), target.dims]

    result = xr.apply_ufunc(
        _gufunc_pchip_roots,
        obj[dim],
        obj,
        target,
        input_core_dims=input_core_dims,
        output_core_dims=[target.dims],
        dask="parallelized",
        output_dtypes=[float],
    )
    result = result.assign_coords({target.dims[0]: target})
    if "target" not in result.dims:
        result = result.expand_dims("target")

    return result
