# These pchip implementations are simplified from those at xyzpy and add dask support
# https://github.com/jcmgray/xyzpy/blob/develop/xyzpy/signal.py

# Note that guvectorized functions must be defined in a module for dask distributed :/
# see https://github.com/numba/numba/issues/4314
# iterative development of a guvectorized function is not possible

# guvectorized functions have out=None, you need to assign to out. no returns allowed!

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
def _gufunc_pchip_roots(x, y, target, out):
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
def _gufunc_pchip(x, y, ix, out=None):
    xy = preprocess_nan_func(x, y, out)
    min_points = 2  # TODO: make this a kwarg
    if xy is None or len(x) < min_points:
        out[:] = np.nan
        return
    x, y = xy

    interpolator = interpolate.PchipInterpolator(x, y, extrapolate=False)
    out[:] = interpolator(ix)


def pchip(obj, dim, ix, core_dim=None):
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

    if isinstance(ix, xr.DataArray):
        ix_da = ix
        if ix_da.ndim == 1:
            core_dim = ix_da.dims[0]
        inferred_core_dim = set(ix_da.dims) - set(obj.dims)
        if len(inferred_core_dim) > 1 and core_dim is None:
            raise ValueError(
                f"Set of dims in `ix` but not in `obj` is not of length 1: {core_dim}, . `core_dim` must be specified explicitly."
            )
        else:
            core_dim = list(inferred_core_dim)[0]
            provided_numpy = False
    else:
        ix_da = xr.DataArray(np.array(ix, ndmin=1), dims="__temp_dim__")
        core_dim = "__temp_dim__"
        provided_numpy = True

    # TODO: unify_chunks

    input_core_dims = [(dim,), (dim,), (core_dim,)]
    if core_dim == dim and not ix.equals(obj[dim]):
        raise ValueError(
            f"core_dim must not be {dim} i.e. not a dimension of the provided DataArray unless the associated coordinates are equal. Please rename this dimension of ix."
        )
    args = (obj[dim], obj, ix_da)

    output_core_dims = [(core_dim,)]

    result = xr.apply_ufunc(
        _gufunc_pchip,
        *args,
        input_core_dims=input_core_dims,
        output_core_dims=output_core_dims,
        dask="parallelized",
        output_dtypes=[float],
        output_sizes={core_dim: ix_da.sizes[core_dim]},
    )

    # TODO: do I really want this?
    # it is convenient and we do want to keep the new coordinates somehow.
    if provided_numpy:
        result = result.rename({core_dim: dim}).assign_coords({f"{dim}": ix_da.values})
    else:
        if ix_da.ndim > 1:
            result = result.assign_coords({f"{dim}_{core_dim}": ix_da})
        else:
            result = result.assign_coords({f"{core_dim}": ix_da.values})

    return result


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
