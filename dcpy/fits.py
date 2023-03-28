import xarray as xr


def fit(curve, x, y, weights=None, doplot=False, **kwargs):
    if weights is None:
        import numpy as np

        weights = np.ones(x.shape)

    if curve == "spline":
        from scipy.interpolate import UnivariateSpline

        spl = UnivariateSpline(x, y, w=weights, check_finite=False)

    else:
        from scipy.optimize import curve_fit

        if curve == "tanh":

            def func(x, y0, X, x0, y1):
                import numpy as np

                return y1 + y0 * np.tanh((x - x0) / X)

        popt, _ = curve_fit(func, x, y, sigma=1 / weights, check_finite=False, **kwargs)

    if doplot:
        import matplotlib.pyplot as plt
        import numpy as np

        # plt.figure()
        plt.cla()
        plt.plot(x, y, "k*")

        xdense = np.linspace(x.min(), x.max(), 200)
        if curve == "spline":
            plt.plot(xdense, spl(xdense), "r-")
        else:
            plt.plot(xdense, func(xdense, *popt), "r-")

    if curve == "spline":
        return spl
    else:
        return popt


def bin_linear_fit(obj, min_count=3, **dim_kwargs):
    """
    Returns slope from linear fit in windows.

    Parameters
    ----------
    obj: DataArray or Dataset
    min_count: int
        Minimum number of data points with which to fit
    **dim_kwargs:
        Mapping of dimension name to window length

    Returns
    -------
    Dataset or DataArray

    Notes
    -----
    1. First coarsens to windows.
    2. Runs polyfit
    3. Assigns nice coordinates
    """
    assert len(dim_kwargs) == 1
    for dim, window in dim_kwargs.items():
        break

    newdim = f"{dim}_"

    coarse = obj.coarsen(**dim_kwargs, boundary="pad").construct(
        {dim: (newdim, "__window__")}
    )
    coarse["__window__"] = obj[dim].data[:window]
    coarse = coarse.where(coarse.count("__window__") > min_count)
    slope = coarse.polyfit("__window__", deg=1).sel(degree=1, drop=True)

    slope[dim] = coarse[dim].mean("__window__")
    slope = slope.swap_dims({newdim: dim})

    if isinstance(obj, xr.DataArray):
        return slope.polyfit_coefficients
    return slope
