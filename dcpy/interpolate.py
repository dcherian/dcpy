# Note that guvectorized functions must be defined in a module for dask distributed :/
# see https://github.com/numba/numba/issues/4314

# guvectorized functions have out=None, you need to assign to out. no returns allowed!
# slightly modified from xyzpy

from xarray import apply_ufunc
from numba import njit, guvectorize, double, int_, jitclass
import numpy as np
from scipy import interpolate
import warnings


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

    interpolator = interpolate.PchipInterpolator(
        x, (y - np.atleast_2d(target).T).T, extrapolate=False, axis=0
    )
    roots = interpolator.roots()
    flattened = roots.ravel()
    for idx, f in enumerate(flattened):
        if f.size > 1:
            warnings.warn("Found multiple roots. Picking the shallowest.", UserWarning)
            flattened[idx] = f[0]
    good = flattened.nonzero()[0]
    out[:] = (
        np.where(np.isin(np.arange(flattened.size), good), flattened, np.nan)
        .astype(x.dtype)
        .reshape(roots.shape)
    )


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


def pchip_roots(obj, dim, target):
    """Interpolate along axis ``dim`` using :func:`scipy.interpolate.pchip`.
    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        The object to interpolate.
    dim : str
        The axis to interpolate along.
    ix : int or array
        If int, interpolate to this many points spaced evenly along the range
        of the original data. If array, interpolate to those points directly.
    Returns
    -------
    new_xobj : xarray.DataArray or xarray.Dataset
    """

    if isinstance(target, (np.ndarray, list)):
        target = xr.DataArray(target, dims="target")

    input_core_dims = [(dim,), (dim,), target.dims]

    result = apply_ufunc(
        _gufunc_pchip_roots,
        obj[dim],
        obj,
        target,
        input_core_dims=input_core_dims,
        output_core_dims=[target.dims],
        dask="parallelized",
        output_dtypes=[float],
    )
    return result.assign_coords({target.dims[0]: target})
