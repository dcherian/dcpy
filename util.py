def ExtractSeason(time, var, season):
    ''' Given a season, return data only for the months in that season
    season can be one of SW, NE, SW->NE, NE→SW or any 3 letter
    month abbreviation.
    '''

    import numpy as np
    from dcpy.util import datenum2datetime

    mask = np.isnan(time)
    time = time[~mask]
    var = var[~mask]

    dates = datenum2datetime(time)
    months = [d.month for d in dates]

    seasonMonths = {'SW':  [5, 6, 7, 8],
                    'SW→NE': [9, 10],
                    'NE':  [11, 12, 1, 2],
                    'NE→SW': [3, 4],
                    'Jan': [1],
                    'Feb': [2],
                    'Mar': [3],
                    'Apr': [4],
                    'May': [5],
                    'Jun': [6],
                    'Jul': [7],
                    'Aug': [8],
                    'Sep': [9],
                    'Oct': [10],
                    'Nov': [11],
                    'Dec': [12]}

    mask = np.asarray([m in seasonMonths[season] for m in months])

    return time[mask], var[mask]


def find_approx(vec, value):
    import numpy as np
    ind = np.nanargmin(np.abs(vec - value))
    return ind


def datenum2datetime(matlab_datenum):
    '''
    Converts matlab datenum to matplotlib datetime.
    '''
    from matplotlib.dates import num2date
    import numpy as np

    python_datetime = num2date(matlab_datenum-367)

    return np.asarray(python_datetime)


def calc95(input, kind='twosided'):

    import numpy as np
    input = np.sort(input)
    if kind is 'twosided':
        interval = input[[np.int(np.floor(0.025*len(input))),
                          np.int(np.ceil(0.975*len(input)))]]
    else:
        interval = input[np.int(np.ceil(0.95*len(input)))]

    return interval


def smooth(x, window_len=11, window='hanning', axis=-1, preserve_nan=True):
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

    window_len = np.int(np.ceil(window_len))
    if np.mod(window_len, 2) < 1e-4:
        window_len = window_len+1

    if window == 'flat':  # moving average
        wnd = np.ones(window_len, 'd')
    else:
        wnd = eval('np.'+window+'(window_len)')

    if x.ndim > 1:
        new_size = np.int32(np.ones((x.ndim, )))
        new_size[axis] = window_len
        new_wnd = np.zeros(new_size)
        new_wnd = np.reshape(new_wnd, (window_len,
                                       np.int32(new_size.prod()/window_len)))
        new_wnd[:, 0] = wnd
        wnd = np.reshape(new_wnd, new_size)

    y = convolve(x, wnd, boundary='fill', normalize_kernel=True,
                 fill_value=np.nan,
                 preserve_nan=preserve_nan, nan_treatment='interpolate')

    return y


def MovingAverage(input, N, decimate=True, min_count=1, **kwargs):
    from bottleneck import move_mean
    import numpy as np

    N = np.int(np.floor(N))

    if N == 1 or N == 0:
        return input
    else:
        y = move_mean(input, window=N, min_count=min_count,
                      **kwargs)
        if decimate:
            return y[N-1:len(y)-N+1:N]
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
        return equalize_hist(1*d, nbins=100000, mask=m)

    x = coords[0]
    y = coords[1]

    mask = np.logical_not(np.logical_or(np.isnan(x), np.isnan(y)))
    hist, xs, ys = np.histogram2d(x[mask], y[mask], bins=bins)
    counts = hist[:, ::-1].T
    transformed = eq_hist(counts, counts != 0)
    span = transformed.max() - transformed.min()
    compressed = np.where(counts != 0,
                          offset+(1.0-offset)*transformed/span,
                          np.nan)
    return xs[:-1], ys[:-1], np.ma.masked_invalid(compressed)


# Print iterations progress
def print_progress(iteration, total, prefix='', suffix='',
                   decimals=1, bar_length=100):
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
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar,
                                            percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')

    sys.stdout.flush()
