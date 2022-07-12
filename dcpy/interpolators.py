import warnings
from functools import partial

import numpy as np
import scipy as sp

# from numba import double, guvectorize, int_, njit
from scipy import interpolate


def preprocess_nan_func(x, y):  # pragma: no cover
    """Pre-process data for a 1d function that doesn't accept nan-values."""
    # strip out nan
    mask = np.isfinite(x) & np.isfinite(y)
    num_nan = np.sum(~mask)

    if x.size - num_nan < 2:
        return None, None
    elif num_nan != 0:
        x = x[mask]
        y = y[mask]
    return x, y


def make_derivative(interpolator):
    def _derivative(x, y, x0):
        """
        Only first-order derivative
        """
        out = np.full_like(x0, np.nan)
        x, y = preprocess_nan_func(x, y)
        if x is None:
            return out

        interpfn = interpolator(x, y)
        out = interpfn.derivative(n=1)(x0)
        out[x0 > x.max()] = np.nan
        out[x0 < x.min()] = np.nan
        return out

    return _derivative


def make_root_finder(interpolator):
    def _roots(x, y, target):
        out = np.full_like(target, np.nan)
        x, y = preprocess_nan_func(x, y)
        if x is None:
            return out

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

        interpfn = interpolator(x, y - target)
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
        out = np.where(
            np.isin(np.arange(flattened.size), good), flattened, np.nan
        ).reshape(roots.shape)

        return out

    return _roots


# https://numba.pydata.org/numba-doc/latest/user/faq.html#can-i-pass-a-function-as-an-argument-to-a-jitted-function
def make_interpolator(interpolator):
    def f(x, y, ix):
        out = np.full_like(ix, np.nan)
        x, y = preprocess_nan_func(x, y)
        if x is None:
            return out

        interpfn = interpolator(x, y)
        out = interpfn(ix)

        # disable extrapolation; needed for UnivariateSpline
        out[ix < x.min()] = np.nan
        out[ix > x.max()] = np.nan

        return out

    return f


def moving_average(series):
    b = sp.signal.get_window(("gaussian", 4), 5, fftbins=False)
    average = sp.ndimage.convolve1d(series, b / b.sum())
    var = sp.ndimage.convolve1d(np.power(series - average, 2), b / b.sum())
    return average, var


def univ_spline(x, y, *args, **kwargs):
    # http://www.nehalemlabs.net/prototype/blog/2014/04/12/how-to-fix-scipys-interpolating-spline-default-behavior/
    _, var = moving_average(y)
    w = 1 / np.sqrt(var)
    return interpolate.UnivariateSpline(x, y, w=w, *args, **kwargs)


_pchip = partial(interpolate.PchipInterpolator, extrapolate=False, axis=-1)
# _gufunc_pchip = make_interpolator(_pchip)


# @guvectorize(
#     [
#         (int_[:], int_[:], double[:], double[:]),
#         (double[:], int_[:], double[:], double[:]),
#         (int_[:], double[:], double[:], double[:]),
#         (double[:], double[:], double[:], double[:]),
#     ],
#     "(n),(n),(m)->(m)",
#     forceobj=True,
#     cache=True,
# )
# def _gufunc_pchip(x, y, ix, out):
#     xy = preprocess_nan_func(x, y, out)
#     min_points = 2  # TODO: make this a kwarg
#     if xy is None or len(x) < min_points:
#         out[:] = np.nan
#         return
#     x, y = xy
#     interpfn = _pchip(x, y)
#     out[:] = interpfn(ix)

#     # disable extrapolation; needed for UnivariateSpline
#     out[ix < x.min()] = np.nan
#     out[ix > x.max()] = np.nan


# @guvectorize(
#     [
#         (int_[:], int_[:], double[:], double[:]),
#         (double[:], int_[:], double[:], double[:]),
#         (int_[:], double[:], double[:], double[:]),
#         (double[:], double[:], double[:], double[:]),
#     ],
#     "(n),(n),(m)->(m)",
#     forceobj=True,
#     cache=True,
# )
# def _gufunc_pchip_roots(x, y, target, out):
#     xy = preprocess_nan_func(x, y, out)
#     if xy is None:
#         out[:] = np.nan
#         return
#     x, y = xy

#     # reshape to [target, ...]
#     target = np.reshape(
#         target,
#         [len(target)]
#         + [
#             1,
#         ]
#         * y.ndim,
#     )
#     y = y[np.newaxis, ...]

#     interpfn = _pchip(x, y - target)
#     roots = interpfn.roots()
#     flattened = roots.ravel()
#     for idx, f in enumerate(flattened):
#         if f.size > 1:
#             warnings.warn(
#                 "Found multiple roots. Picking the first one. This will depend on the ordering of `dim`",
#                 UserWarning,
#             )
#             flattened[idx] = f[0]
#     good = flattened.nonzero()[0]
#     out[:] = np.where(
#         np.isin(np.arange(flattened.size), good), flattened, np.nan
#     ).reshape(roots.shape)


# @guvectorize(
#     [
#         (int_[:], int_[:], double[:], double[:]),
#         (double[:], int_[:], double[:], double[:]),
#         (int_[:], double[:], double[:], double[:]),
#         (double[:], double[:], double[:], double[:]),
#     ],
#     "(n),(n),(m)->(m)",
#     forceobj=True,
#     cache=True,
# )
# def _gufunc_spline(x, y, ix, out):
#     xy = preprocess_nan_func(x, y, out)
#     min_points = 2  # TODO: make this a kwarg
#     if xy is None or len(x) < min_points:
#         out[:] = np.nan
#         return
#     x, y = xy
#     interpfn = univ_spline(x, y)
#     out[:] = interpfn(ix)

#     # disable extrapolation; needed for UnivariateSpline
#     out[ix < x.min()] = np.nan
#     out[ix > x.max()] = np.nan


# @guvectorize(
#     [
#         (int_[:], int_[:], double[:], double[:]),
#         (double[:], int_[:], double[:], double[:]),
#         (int_[:], double[:], double[:], double[:]),
#         (double[:], double[:], double[:], double[:]),
#     ],
#     "(n),(n),(m)->(m)",
#     forceobj=True,
#     cache=True,
# )
# def _gufunc_spline_roots(x, y, target, out):
#     xy = preprocess_nan_func(x, y, out)
#     if xy is None:
#         out[:] = np.nan
#         return
#     x, y = xy

#     # reshape to [target, ...]
#     target = np.reshape(
#         target,
#         [len(target)]
#         + [
#             1,
#         ]
#         * y.ndim,
#     )
#     y = y[np.newaxis, ...]

#     interpfn = univ_spline(x, y - target)
#     roots = interpfn.roots()
#     flattened = roots.ravel()
#     for idx, f in enumerate(flattened):
#         if f.size > 1:
#             warnings.warn(
#                 "Found multiple roots. Picking the first one. This will depend on the ordering of `dim`",
#                 UserWarning,
#             )
#             flattened[idx] = f[0]
#     good = flattened.nonzero()[0]
#     out[:] = np.where(
#         np.isin(np.arange(flattened.size), good), flattened, np.nan
#     ).reshape(roots.shape)


# @guvectorize(
#     [
#         (int_[:], int_[:], double[:], double[:]),
#         (double[:], int_[:], double[:], double[:]),
#         (int_[:], double[:], double[:], double[:]),
#         (double[:], double[:], double[:], double[:]),
#     ],
#     "(n),(n),(m)->(m)",
#     forceobj=True,
# )
# def _gufunc_spline_der(x, y, x0, out):
#     """
#     Only first-order derivative
#     """
#     xy = preprocess_nan_func(x, y, out)
#     if xy is None:
#         out[:] = np.nan
#         return
#     x, y = xy

#     interpfn = univ_spline(x, y)
#     out[:] = interpfn.derivative(n=1)(x0)
#     out[x0 > x.max()] = np.nan
#     out[x0 < x.min()] = np.nan


# @guvectorize(
#     [
#         (int_[:], int_[:], double[:], double[:]),
#         (double[:], int_[:], double[:], double[:]),
#         (int_[:], double[:], double[:], double[:]),
#         (double[:], double[:], double[:], double[:]),
#     ],
#     "(n),(n),(m)->(m)",
#     forceobj=True,
#     cache=True,
# )
# def _gufunc_spline(x, y, ix, out):
#     xy = preprocess_nan_func(x, y, out)
#     min_points = 2  # TODO: make this a kwarg
#     if xy is None or len(x) < min_points:
#         out[:] = np.nan
#         return
#     x, y = xy
#     interpfn = univ_spline(x, y)
#     out[:] = interpfn(ix)

#     # disable extrapolation; needed for UnivariateSpline
#     out[ix < x.min()] = np.nan
#     out[ix > x.max()] = np.nan


# @guvectorize(
#     [
#         (int_[:], int_[:], double[:], double[:]),
#         (double[:], int_[:], double[:], double[:]),
#         (int_[:], double[:], double[:], double[:]),
#         (double[:], double[:], double[:], double[:]),
#     ],
#     "(n),(n),(m)->(m)",
#     forceobj=True,
#     cache=True,
# )
# def _gufunc_spline_roots(x, y, target, out):
#     xy = preprocess_nan_func(x, y, out)
#     if xy is None:
#         out[:] = np.nan
#         return
#     x, y = xy

#     # reshape to [target, ...]
#     target = np.reshape(
#         target,
#         [len(target)]
#         + [
#             1,
#         ]
#         * y.ndim,
#     )
#     y = y[np.newaxis, ...]

#     interpfn = univ_spline(x, y - target)
#     roots = interpfn.roots()
#     flattened = roots.ravel()
#     for idx, f in enumerate(flattened):
#         if f.size > 1:
#             warnings.warn(
#                 "Found multiple roots. Picking the first one. This will depend on the ordering of `dim`",
#                 UserWarning,
#             )
#             flattened[idx] = f[0]
#     good = flattened.nonzero()[0]
#     out[:] = np.where(
#         np.isin(np.arange(flattened.size), good), flattened, np.nan
#     ).reshape(roots.shape)


# @guvectorize(
#     [
#         (int_[:], int_[:], double[:], double[:]),
#         (double[:], int_[:], double[:], double[:]),
#         (int_[:], double[:], double[:], double[:]),
#         (double[:], double[:], double[:], double[:]),
#     ],
#     "(n),(n),(m)->(m)",
#     forceobj=True,
# )
# def _gufunc_spline_der(x, y, x0, out):
#     """
#     Only first-order derivative
#     """
#     xy = preprocess_nan_func(x, y, out)
#     if xy is None:
#         out[:] = np.nan
#         return
#     x, y = xy

#     interpfn = univ_spline(x, y)
#     out[:] = interpfn.derivative(n=1)(x0)
#     out[x0 > x.max()] = np.nan
#     out[x0 < x.min()] = np.nan
