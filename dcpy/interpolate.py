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

from . import interpolators

from .interpolators import (
    _gufunc_pchip,
    _gufunc_pchip_roots,
    preprocess_nan_func,
    univ_spline,
)


# _pchip = partial(interpolate.PchipInterpolator, extrapolate=False, axis=-1)
# _gufunc_pchip = interpolators.make_interpolator(_pchip)
# _gufunc_pchip_roots = interpolators.make_root_finder(_pchip)

# _gufunc_spline = interpolators.make_interpolator(interpolators.univ_spline)
# _gufunc_spline_roots = interpolators.make_root_finder(interpolators.univ_spline)
# _gufunc_spline_der = interpolators.make_derivative(interpolators.univ_spline)


@guvectorize(
    [
        (int_[:], int_[:], double[:], double[:]),
        (double[:], int_[:], double[:], double[:]),
        (int_[:], double[:], double[:], double[:]),
        (double[:], double[:], double[:], double[:]),
    ],
    "(n),(n),(m)->(m)",
    forceobj=True,
    cache=True,
)
def _gufunc_spline(x, y, ix, out):
    xy = preprocess_nan_func(x, y, out)
    min_points = 2  # TODO: make this a kwarg
    if xy is None or len(x) < min_points:
        out[:] = np.nan
        return
    x, y = xy
    interpfn = univ_spline(x, y)
    out[:] = interpfn(ix)

    # disable extrapolation; needed for UnivariateSpline
    out[ix < x.min()] = np.nan
    out[ix > x.max()] = np.nan


@guvectorize(
    [
        (int_[:], int_[:], double[:], double[:]),
        (double[:], int_[:], double[:], double[:]),
        (int_[:], double[:], double[:], double[:]),
        (double[:], double[:], double[:], double[:]),
    ],
    "(n),(n),(m)->(m)",
    forceobj=True,
    cache=True,
)
def _gufunc_spline_roots(x, y, target, out):
    xy = preprocess_nan_func(x, y, out)
    if xy is None:
        out[:] = np.nan
        return
    x, y = xy

    # reshape to [target, ...]
    target = np.reshape(
        target,
        [len(target)]
        + [
            1,
        ]
        * y.ndim,
    )
    y = y[np.newaxis, ...]

    interpfn = univ_spline(x, y - target)
    roots = interpfn.roots()
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
def _gufunc_spline_der(x, y, x0, out):
    """
    Only first-order derivative
    """
    xy = preprocess_nan_func(x, y, out)
    if xy is None:
        out[:] = np.nan
        return
    x, y = xy

    interpfn = univ_spline(x, y)
    out[:] = interpfn.derivative(n=1)(x0)
    out[x0 > x.max()] = np.nan
    out[x0 < x.min()] = np.nan


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
        dask_gufunc_kwargs=dict(output_sizes={core_dim: ix_da.sizes[core_dim]}),
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


def _roots(obj, dim, target, _root_finder):
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
        _root_finder,
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


class Interpolator:
    def __init__(self, interpolator, root_finder, derivative, obj, dim):
        self._interp_gufunc = interpolator
        self._roots_gufunc = root_finder
        self._der_gufunc = derivative
        self.obj = obj
        self.dim = dim

    def smooth(self, *args, **kwargs):
        return self.interp(ix=self.obj[self.dim], *args, **kwargs)

    def interp(self, ix, core_dim=None, *args, **kwargs):
        return _interpolator(
            obj=self.obj,
            dim=self.dim,
            ix=ix,
            interp_gufunc=self._interp_gufunc,
            *args,
            **kwargs,
        )

    def roots(self, target=0.0):
        if self._roots_gufunc is not None:
            return _roots(
                obj=self.obj,
                dim=self.dim,
                target=target,
                _root_finder=self._roots_gufunc,
            )
        else:
            raise NotImplementedError(
                f"Root finding not implemented for {str(self._interp_gufunc)} yet"
            )

    def derivative(self, x0):
        if self._der_gufunc is not None:
            return _interpolator(
                obj=self.obj,
                dim=self.dim,
                ix=x0,
                interp_gufunc=self._der_gufunc,
            )
        else:
            raise NotImplementedError(
                f"Derviatives not implemented for {str(self._interp_gufunc)} yet"
            )

        pass


def PchipInterpolator(obj, dim):
    return Interpolator(_gufunc_pchip, _gufunc_pchip_roots, None, obj, dim)


def pchip(obj, dim, ix, core_dim=None, *args, **kwargs):
    interpolator = PchipInterpolator(obj, dim)
    return interpolator.interp(ix=ix, core_dim=core_dim, *args, **kwargs)


def pchip_roots(obj, dim, target=0.0):
    interpolator = PchipInterpolator(obj, dim)
    return interpolator.roots(target=target)


def UnivariateSpline(obj, dim):
    return Interpolator(
        _gufunc_spline,
        _gufunc_spline_roots,
        _gufunc_spline_der,
        obj,
        dim,
    )


def pchip_fillna(obj, dim):
    return pchip(obj, dim, obj[dim])


@guvectorize(
    [(double[:], double[:], double[:], double[:], int_[:])],
    "(n), (n), (m) -> (m), (m)",
    nopython=True,
)
def remap(values, zeuc, edges, out=None, count=None):
    func = np.nanmean

    out[:] = np.nan
    count[:] = 0

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
                count[np.int(ii) - 1] = np.sum(mask)


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

    remapped, counts = xr.apply_ufunc(
        remap,
        data,
        data[new_coord],
        edges,
        input_core_dims=[[old_coord], [old_coord], ["__temp_dim__"]],
        output_core_dims=[[new_coord], [new_coord]],
        exclude_dims=set((old_coord,)),
        dask="parallelized",
        output_dtypes=[float, int],
        dask_gufunc_kwargs=dict(output_sizes={new_coord: len(edges)}),
        # kwargs={"func": reduce_func}  # TODO: add support for reduce_func
    )
    remapped.coords["counts"] = counts
    remapped.counts.attrs["description"] = "number of values in the bin"
    remapped = remapped.isel({new_coord: slice(-1)})
    remapped[new_coord] = new_1d_coord

    return remapped


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
def _gufunc_pchip_roots_old(x, y, target, out):
    xy = preprocess_nan_func(x, y, out)
    if xy is None:
        out[:] = np.nan
        return
    x, y = xy

    # reshape to [target, ...]
    target = np.reshape(
        target,
        [len(target)]
        + [
            1,
        ]
        * y.ndim,
    )
    y = y[np.newaxis, ...]

    interpolator = interpolate.PchipInterpolator(
        x,
        y - target,
        extrapolate=False,
        axis=-1,
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


def pchip_roots_old(obj, dim, target):
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
        _gufunc_pchip_roots_old,
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
