# These pchip implementations are simplified from those at xyzpy and add dask support
# https://github.com/jcmgray/xyzpy/blob/develop/xyzpy/signal.py

# Note that guvectorized functions must be defined in a module for dask distributed :/
# see https://github.com/numba/numba/issues/4314
# iterative development of a guvectorized function is not possible

# guvectorized functions have out=None, you need to assign to out. no returns allowed!

import warnings
from functools import partial
from typing import Any, Iterable

import numpy as np
import scipy as sp
from numba import double, guvectorize, int_, njit
from scipy import interpolate

import xarray as xr


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


# https://numba.pydata.org/numba-doc/latest/user/faq.html#can-i-pass-a-function-as-an-argument-to-a-jitted-function
def make_interpolator(interpolator):
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
    def f(x, y, ix, out):
        xy = preprocess_nan_func(x, y, out)
        min_points = 2  # TODO: make this a kwarg
        if xy is None or len(x) < min_points:
            out[:] = np.nan
            return
        x, y = xy
        interpfn = interpolator(x, y)
        out[:] = interpfn(ix)

        # disable extrapolation; needed for UnivariateSpline
        out[ix < x.min()] = np.nan
        out[ix > x.max()] = np.nan

    return f


def _interpolator(obj, dim, ix, core_dim=None, interp_gufunc=None, *args, **kwargs):
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
        ix_da = ix.copy()
        if ix_da.ndim == 1:
            core_dim = ix_da.dims[0]
        else:
            inferred_core_dim = set(ix_da.dims) - set(obj.dims)
            if len(inferred_core_dim) == 0 and core_dim is None:
                raise ValueError(
                    "Could not infer the core interpolation dimension. Please explicitly pass core_dim."
                )
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

    if core_dim == dim and not ix.equals(obj[dim]):
        ix_da = ix_da.rename({dim: "__temp_dim__"})
        core_dim = "__temp_dim__"

    extra_args = (obj[dim], obj, ix_da)
    input_core_dims = [(dim,), (dim,), (core_dim,)]
    output_core_dims = [(core_dim,)]

    result = xr.apply_ufunc(
        interp_gufunc,
        *extra_args,
        *args,
        input_core_dims=input_core_dims,
        output_core_dims=output_core_dims,
        dask="parallelized",
        output_dtypes=[float],
        output_sizes={core_dim: ix_da.sizes[core_dim]},
        kwargs=kwargs,
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
    if "__temp_dim__" in result.dims:
        result = result.rename({"__temp_dim__": dim})

    return result


_gufunc_pchip = make_interpolator(
    partial(interpolate.PchipInterpolator, extrapolate=False)
)
pchip = partial(_interpolator, interp_gufunc=_gufunc_pchip)
pchip.__doc__ = "Uses PCHIP interpolator\n\n" + pchip.__doc__


# http://www.nehalemlabs.net/prototype/blog/2014/04/12/how-to-fix-scipys-interpolating-spline-default-behavior/
def moving_average(series):
    b = sp.signal.get_window(("gaussian", 4), 5, fftbins=False)
    average = sp.ndimage.convolve1d(series, b / b.sum())
    var = sp.ndimage.convolve1d(np.power(series - average, 2), b / b.sum())
    return average, var


def univ_spline(x, y):
    _, var = moving_average(y)
    w = 1 / np.sqrt(var)
    return interpolate.UnivariateSpline(x, y, w=w)


spline = partial(_interpolator, interp_gufunc=make_interpolator(univ_spline))
spline.__doc__ = "Uses UnivariateSpline interpolator\n\n" + spline.__doc__


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
    # result = result.assign_coords({target.dims[0]: target})
    if "target" not in result.dims:
        result = result.expand_dims("target")

    return result


@guvectorize(
    [(double[:], double[:], double[:], double[:])],
    "(n), (n), (m) -> (m)",
    nopython=True,
)
def remap(values, zeuc, edges, out=None):
    func = np.nanmean

    out[:] = np.nan
    idx = np.digitize(zeuc, edges)
    idx = np.where(~np.isnan(zeuc), idx, np.nan)

    minz = edges.min()
    maxz = edges.max()

    for ii in np.unique(idx):
        # 1. handle bunch of values in same bin
        # 2. should also take care of NaN
        # 3. from digitize docstring
        #         If values in x are beyond the bounds of bins,
        #         0 or len(bins) is returned as appropriate.
        if ~np.isnan(ii):
            mask = (idx == ii) & (zeuc >= minz) & (zeuc <= maxz)
            if np.any(mask):
                out[np.int(ii) - 1] = func(values[mask])


def bin_to_new_coord(data, old_coord, new_coord, edges, reduce_func=None):
    """
    Bin to a new 1D coordinate.
    """

    if not isinstance(old_coord, str) or old_coord not in data.dims:
        raise ValueError(
            f"old_coord must be the name of an existing dimension in data."
            f"Expected one of {data.dims}. Received {old_coord}."
        )

    if not isinstance(new_coord, str) or new_coord not in data.coords:
        raise ValueError(
            f"old_coord must be the name of an existing coordinate variable."
            f"Expected one of {set(data.coords)}. Received {new_coord}."
        )

    if reduce_func is not None:
        raise ValueError("reduce_func support has not been implemented yet")

    new_1d_coord = xr.DataArray((edges[:-1] + edges[1:]) / 2, dims=(new_coord,))

    remapped = xr.apply_ufunc(
        remap,
        data,
        data[new_coord],
        edges,
        input_core_dims=[[old_coord], [old_coord], ["__temp_dim__"]],
        output_core_dims=[[new_coord]],
        exclude_dims=set((old_coord,)),
        dask="parallelized",
        # vectorize=True,  # TODO: guvectorize instead
        output_dtypes=[float],
        output_sizes={new_coord: len(edges)},
        # kwargs={"func": reduce_func}  # TODO: add support for reduce_func
    ).isel({new_coord: slice(-1)})
    remapped[new_coord] = new_1d_coord

    return remapped
