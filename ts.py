import numpy as np
import scipy.signal as signal
import scipy.fftpack as fftpack

def FindLargestSegment(input):
    import numpy as np

    start, stop = FindSegments(input)
    GapLength = stop-start+1
    imax = np.argmax(GapLength)

    return start[imax], stop[imax]

def FindSegments(input):
    '''
      Finds and return valid index ranges for the input time series.
      Input:
            input - input time series
      Output:
            start - starting indices of valid ranges
            stop  - ending indices of valid ranges
    '''

    import numpy as np

    NotNans = np.double(~np.isnan(input))
    edges = np.diff(NotNans)
    start = np.where(edges == 1)[0]
    stop = np.where(edges == -1)[0]

    if start.size == 0 and stop.size == 0:
        start = np.array([0])
        stop = np.array([len(input)-1])

    else:
        start = start + 1
        if ~np.isnan(input[0]):
            start = np.insert(start, 0, 0)

            if ~np.isnan(input[-1]):
                stop = np.append(stop, len(input)-1)

    return start, stop

def CenteredFFT(input, dt=1.0):
    N = len(input)

    # Generate frequency index
    if np.mod(N, 2) == 0:
        m = np.arange(-N/2, N/2)
    else:
        m = np.arange(-(N-1)/2, (N-1)/2+1)

    freq = m/(N*dt)
    X = fftpack.rfft(input)
    X = fftpack.fftshift(X)

    return X, freq

def AliasFreq(f0, dt):

    fs = 1/dt

    lower = np.ceil(f0/fs - 0.5)
    upper = np.floor(f0/fs + 0.5)

    for n in [lower, upper, -lower, -upper]:
        fa = f0 + n/dt
        if fa <= 1/(2*dt) and fa >= -1/(2*dt):
            return np.abs(fa)

    raise ValueError('No integer found for aliasing')

def ConfChi2(alpha, dof):
    import numpy as np
    from scipy.stats import chi2

    return np.sort(dof/np.array(chi2.interval(1-alpha, dof)))

def SpectralDensity(input, dt=1, nsmooth=5):
    """ Calculates spectral density for longest valid segment
        Direct translation of Tom's spectrum_band_avg.
        Always applies a Hann window.
        Input:
            input : variable whos spectral density you want
            dt : Δtime
            nsmooth : number of frequency bands to average over

        Returns:
            S : estimated spectral density
            f : frequency bands
         conf : 95% confidence interval for 2*nsmooth dof
    """
    import scipy.signal as signal
    import dcpy.util
    import numpy as np

    start, stop = FindLargestSegment(input)
    var = input[start:stop]

    N = len(var)
    T = N*dt
    window = signal.hann(N)
    # variance correction
    window /= np.sqrt(np.sum(window**2)/N)

    var -= var.mean()
    var = var * window

    [Y, freq] = CenteredFFT(var, dt)
    Y = Y[freq > 0]
    freq = freq[freq > 0]
    YY_raw = 2*T/N**2 * Y * np.conj(Y)

    S = dcpy.util.MovingAverage(YY_raw, nsmooth)
    f = dcpy.util.MovingAverage(freq, nsmooth)
    conf = ConfChi2(0.05, 2*nsmooth)

    return S, f, conf, var, window

def HighPassButter(input, freq):
    import scipy.signal as signal

    b, a = signal.butter(1, freq/(1/2), btype='high')

    return GappyFilter(input, b, a, 10)

def GappyFilter(input, b, a, num_discard=None):
    import scipy.signal as signal

    segstart,segend = FindSegments(input)
    out = np.empty(input.shape) * np.nan
    for index, start in np.ndenumerate(segstart):
        stop = segend[index]
        out[start:stop] = signal.lfilter(b, a, input[start:stop])
        if num_discard is not None:
            out[start:start+num_discard] = np.nan
            out[stop-num_discard:stop] = np.nan

    return out

def HighPassAndPlot(input, CutoffFreq, titlestr=None):

    start, stop = FindLargestSegment(input)
    filtered = HighPassButter(input, CutoffFreq)

    f, InputSpec = SpectralDensity(input, 10)
    plt.loglog(f, InputSpec, label='input data')

    f, FiltSpec = SpectralDensity(filtered, 10)
    plt.loglog(f, FiltSpec, label='high pass')

    plt.axvline(CutoffFreq, color='gray', zorder=-20)
    plt.ylabel('Spectral density')
    plt.xlabel('Frequency')
    plt.title(titlestr)
    plt.legend()

    return filtered