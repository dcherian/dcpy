import numpy as np
import pandas as pd
import scipy as sp

import xarray as xr


def to_uniform_grid(data, coord, dim, new_coord):
    """
    Interpolate from multidimensional coordinate to 1D coordinate.

    Inputs
    ------

    data: DataArray, Dataset
        Input data to interpolate

    coord : str
        Name of the multidimensional coordinate that we will be interpolating to.
        Must exist in data

    dim : str
        Dimension name whose size will change after interpolation

    new_coord : np.ndarray, DataArray
        Uniform coordinate to interpolate to
    """

    if coord not in data:
        raise ValueError(f"{coord} is not in the provided dataset.")

    if isinstance(new_coord, np.ndarray):
        new_coord = xr.DataArray(new_coord, dims=[coord], name=coord)

    def _wrap_interp(x, y, newy):
        f = sp.interpolate.interp1d(
            x.squeeze(), y.squeeze(), bounds_error=False, fill_value=np.nan
        )
        return f(newy)

    dim = set([dim])

    result = xr.apply_ufunc(
        _wrap_interp,
        data[coord],
        data,
        new_coord,
        input_core_dims=[dim, dim, [coord]],
        output_core_dims=[[coord]],  # order is important
        exclude_dims=dim,  # since size of dimension is changing
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64],
    )

    result[coord] = new_coord

    return result


def split_to_dataset(obj, dim, maxlen):
    grouped = obj.groupby(dim)

    grouped_dim = grouped._group_dim
    idim = "i" + grouped_dim
    reshaped = []
    for label, group in grouped:
        time = (
            group.time.reset_coords(drop=True)
            .drop(grouped_dim)
            .rename({grouped_dim: idim})
            .expand_dims(dim)
        )

        reshaped.append(
            group.reset_coords()
            .drop([dim, grouped_dim])
            .rename({grouped_dim: idim})
            .assign({grouped_dim: time, idim: np.arange(len(group[grouped_dim]))})
            .reindex(itime=np.arange(maxlen))
        )

    dataset = xr.concat(reshaped, dim=dim)
    dataset[dim] = grouped._unique_coord

    return dataset


def ExtractSeason(time, var, season):
    """Given a season, return data only for the months in that season
    season can be one of SW, NE, SW->NE, NE→SW or any 3 letter
    month abbreviation.
    """

    import numpy as np
    from dcpy.util import datenum2datetime

    mask = np.isnan(time)
    time = time[~mask]
    var = var[~mask]

    dates = datenum2datetime(time)
    months = [d.month for d in dates]

    seasonMonths = {
        "SW": [5, 6, 7, 8],
        "SW→NE": [9, 10],
        "NE": [11, 12, 1, 2],
        "NE→SW": [3, 4],
        "Jan": [1],
        "Feb": [2],
        "Mar": [3],
        "Apr": [4],
        "May": [5],
        "Jun": [6],
        "Jul": [7],
        "Aug": [8],
        "Sep": [9],
        "Oct": [10],
        "Nov": [11],
        "Dec": [12],
    }

    mask = np.asarray([m in seasonMonths[season] for m in months])

    return time[mask], var[mask]


def find_approx(vec, value):
    import numpy as np

    ind = np.nanargmin(np.abs(vec - value))
    return ind


def dt64_to_datenum(dt64):
    import matplotlib.dates as mdt
    import datetime as pdt

    return mdt.date2num(dt64.astype("M8[s]").astype(pdt.datetime))


def mdatenum2dt64(dnum):
    import numpy as np

    return (-86400 + dnum * 86400).astype("timedelta64[s]") + np.datetime64(
        "0001-01-01"
    )


def datenum2datetime(matlab_datenum):
    """
    Converts matlab datenum to matplotlib datetime.
    """
    import numpy as np

    python_datetime = (
        (-86400 + matlab_datenum * 86400).astype("timedelta64[s]")
        + np.datetime64("0000-01-01")
    ).astype("datetime64[ns]")
    # python_datetime = num2date(matlab_datenum-366)

    return np.asarray(python_datetime)


def calc95(input, kind="twosided"):

    import numpy as np

    input = np.sort(input)
    if kind == "twosided":
        interval = input[
            [np.int(np.floor(0.025 * len(input))), np.int(np.ceil(0.975 * len(input)))]
        ]
    else:
        interval = input[np.int(np.ceil(0.95 * len(input)))]

    return interval


def smooth(x, window_len=11, window="hanning", axis=-1, preserve_nan=True):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window
    with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are
    minimized in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window;
                    should be an odd integer
        window: the type of window from 'flat', 'hanning',
                'hamming', 'bartlett', 'blackman'
        flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    TODO: the window parameter could be the window itself if an
          array instead of a string
    NOTE: length(output) != length(input), to correct this:
          return y[(window_len/2-1):-(window_len/2)]
          instead of just y.
    """

    import numpy as np
    from astropy.convolution import convolve

    if x.shape[axis] < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window == "hann":
        window = "hanning"

    window_len = np.int(np.ceil(window_len))
    if np.mod(window_len, 2) < 1e-4:
        window_len = window_len + 1

    if window == "flat":  # moving average
        wnd = np.ones(window_len, "d")
    else:
        wnd = eval("np." + window + "(window_len)")

    if x.ndim > 1:
        new_size = np.int32(np.ones((x.ndim,)))
        new_size[axis] = window_len
        new_wnd = np.zeros(new_size)
        new_wnd = np.reshape(
            new_wnd, (window_len, np.int32(new_size.prod() / window_len))
        )
        new_wnd[:, 0] = wnd
        wnd = np.reshape(new_wnd, new_size)

    y = convolve(
        x,
        wnd,
        boundary="fill",
        normalize_kernel=True,
        fill_value=np.nan,
        preserve_nan=preserve_nan,
        nan_treatment="interpolate",
    )

    return y


def MovingAverage(vin, N, dim=None, decimate=True, min_count=1, **kwargs):
    from bottleneck import move_mean
    import numpy as np

    N = np.int(np.floor(N))

    if N == 1 or N == 0:
        return vin
    else:
        y = move_mean(vin, window=N, min_count=min_count, **kwargs)
        if decimate:
            return y[N - 1 : len(y) - N + 1 : N]
        else:
            return y


def BinEqualizeHist(coords, bins=10, offset=0.0, label=None):
    """
    Given a set of coordinates, bins them into a 2d histogram grid
    of the specified size, and optionally transforms the counts
    and/or compresses them into a visible range starting at a
    specified offset between 0 and 1.0.
    """
    from skimage.exposure import equalize_hist
    import numpy as np

    def eq_hist(d, m):
        return equalize_hist(1 * d, nbins=100000, mask=m)

    x = coords[0]
    y = coords[1]

    mask = np.logical_not(np.logical_or(np.isnan(x), np.isnan(y)))
    hist, xs, ys = np.histogram2d(x[mask], y[mask], bins=bins)
    counts = hist[:, ::-1].T
    transformed = eq_hist(counts, counts != 0)
    span = transformed.max() - transformed.min()
    compressed = np.where(
        counts != 0, offset + (1.0 - offset) * transformed / span, np.nan
    )
    compressed = np.flipud(compressed)
    return xs[:-1], ys[:-1], np.ma.masked_invalid(compressed)


# Print iterations progress
def print_progress(iteration, total, prefix="", suffix="", decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals
                                  in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """

    import sys

    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = "█" * filled_length + "-" * (bar_length - filled_length)

    sys.stdout.write("\r%s |%s| %s%s %s" % (prefix, bar, percents, "%", suffix)),

    if iteration == total:
        sys.stdout.write("\n")

    sys.stdout.flush()


def rms(da, axis="time"):
    return np.sqrt((np.abs(da) ** 2).mean(axis))


def ms(da, axis="time"):
    return (np.abs(da) ** 2).mean(axis)


def calc_iso_surface(data, value, zs, interp_order=3, power_parameter=0.5):
    """
    Estimate location of isosurface along 3rd dimension.

    Parameters
    ==========

    data: ndarray
        3d array

    value: float
        Isosurface value to search for

    zs : 1D array,
        Co-ordinate for axis along which we are searching

    interp_order: int, optional
        Max polynomial order.

    power_parameter: float, optional

    References
    ==========
    1. https://stackoverflow.com/questions/13627104/using-numpy-scipy-to-calculate-iso-surface-from-3d-array
    """

    if interp_order < 1:
        interp_order = 1

    dist = (data - value) ** 2
    arg = np.argsort(dist, axis=2)
    dist.sort(axis=2)
    w_total = 0.0
    z = np.zeros(data.shape[:2], dtype=float)
    for i in range(int(interp_order)):
        zi = np.take(zs, arg[:, :, i])
        valuei = dist[:, :, i]
        wi = 1 / valuei
        np.clip(wi, 0, 1.0e6, out=wi)  # avoiding overflows
        w_total += wi ** power_parameter
        z += zi * wi ** power_parameter

    z /= w_total

    return z


def interp_zero_crossing(da, debug=False):
    def crossings_nonzero_all(data):
        pos = data > 0
        npos = ~pos
        return ((pos[:-1] & npos[1:]) | (npos[:-1] & pos[1:])).nonzero()[0]

    newtime = pd.date_range(da.time[0].values, da.time[-1].values, freq="1min")
    interped = da.compute().interp(time=newtime, method="cubic")

    idx = crossings_nonzero_all(interped.values)
    coords = interped.time[idx]

    if debug:
        interped.plot(marker=".")
        dcpy.plots.linex(coords, zorder=10)
        plt.axhline(0)

    return coords


def index_unindexed_dims(obj):
    """
    Adds integer indexes to any unindexed dimensions in obj.
    """
    obj = obj.copy()
    dims = set(obj.dims) - set(obj.indexes)
    for dim in dims:
        obj = obj.assign_coords({dim: np.arange(obj.sizes[dim])})
    return obj


def avg1(da, dim):

    return da.isel({dim: slice(-1)}).copy(
        data=(
            da.isel({dim: slice(-1)}).data
            + da.isel(
                {
                    dim: slice(
                        1,
                    )
                }
            ).data
        )
        / 2
    )


def latlon_to_distance(lat, lon, *, central_lat, central_lon):
    """ Returns distance relative to central_lat, central_lon """

    lat0 = central_lat * np.pi / 180
    lon0 = central_lon * np.pi / 180
    lat = lat * np.pi / 180
    lon = lon * np.pi / 180
    r = 6378137

    dlat = lat - lat0
    dlon = lon - lon0

    a = np.sin(dlat / 2) ** 2
    b = np.cos(lat0) * np.cos(lat) * np.sin(dlon / 2) ** 2
    d = 2 * r * np.arcsin(np.sqrt(a + b)).clip(max=1)

    d.attrs["long_name"] = f"Distance from ({central_lon}°, {central_lat}°)"
    d.attrs["units"] = "m"
    return d


def latlon_to_xy(lat, lon):
    R = 6371000  # m
    x = R * np.cos(lat * np.pi / 180) * np.cos(lon * np.pi / 180)
    y = R * np.cos(lat * np.pi / 180) * np.sin(lon * np.pi / 180)

    x.attrs["axis"] = "X"
    y.attrs["axis"] = "Y"

    x.attrs["units"] = "m"
    y.attrs["units"] = "m"

    return x, y


def infer_percentile(data, dim, value, debug=False):
    """
    Infers percentile of `value` in `data` along dimension `dim`.
    """

    data = data.compute()

    if data.isnull().all():
        inferred_q = data.isel({dim: 0}, drop=True) * np.nan
    else:
        closest = np.abs(data - value).fillna(1e20).argmin(dim)
        N = data.notnull().sum(dim)
        inferred_q = data.rank(dim).isel({dim: closest}, drop=True) / N

        inferred_q = inferred_q.where(data.min(dim) < value, np.nan)
        inferred_q = inferred_q.where(data.max(dim) > value, np.nan)

    if debug:
        actual = data.quantile(q=inferred_q.data, dim=dim).values
        print(f"inferred_q: {inferred_q.data} | actual: {actual} vs expect:{value}")

    return inferred_q.reset_coords(drop=True)


def slice_like(this, other):
    dims = set(this.dims) & set(other.dims)
    slicer = {dim: slice(other[dim][0], other[dim][-1]) for dim in dims}
    return this.sel(slicer)
