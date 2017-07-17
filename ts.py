import numpy as np
import scipy.signal as signal
import scipy.fftpack as fftpack


def FindLargestSegment(input):

    start, stop = FindSegments(input)
    GapLength = stop-start+1
    imax = np.argmax(GapLength)

    return start[imax], stop[imax]


def FindSegments(var):
    '''
      Finds and return valid index ranges for the input time series.
      Input:
            var - input time series
      Output:
            start - starting indices of valid ranges
            stop  - ending indices of valid ranges
    '''

    NotNans = np.double(~np.isnan(var))
    edges = np.diff(NotNans)
    start = np.where(edges == 1)[0]
    stop = np.where(edges == -1)[0]

    if start.size == 0 and stop.size == 0:
        start = np.array([0])
        stop = np.array([len(var)-1])

    else:
        start = start + 1
        if ~np.isnan(var[0]):
            start = np.insert(start, 0, 0)

        if ~np.isnan(var[-1]):
            stop = np.append(stop, len(var)-1)

    return start, stop


def PlotSpectrum(var, ax=None, dt=1, nsmooth=5,
                 SubsetLength=None, breakpts=[], **kwargs):

    import matplotlib.pyplot as plt

    start, stop = FindLargestSegment(var)
    S, f, conf = SpectralDensity(var, dt, nsmooth, SubsetLength,
                                 breakpts=breakpts)

    if ax is None:
        ax = plt.gca()

    hdl = ax.loglog(f, S, **kwargs)
    if len(conf) > 2:
        ax.fill_between(f, conf[:, 0], conf[:, 1],
                        color=hdl[0].get_color(), alpha=0.3)

    return hdl


def synthetic(N, dt, α, β):
    '''
    Generate time series with spectrum S = α ω^β

    Input:
        N : number of points
        dt : time interval
        α, β : spectrum parameters

    Output:
        yfilt : time series with desired spectral shape

    Ack:
       Copied from Tom Farrar's synthetic_timeseries_known_spectrum
    '''

    from numpy import sqrt

    y = np.random.randn(N)

    [Y, freq] = CenteredFFT(y, dt)

    Yfilt = sqrt(α) * sqrt(1./(2*dt)) \
        * (np.abs(freq)**(β/2)) * Y

    ff = np.where(freq == 0)
    Yfilt[ff] = 0
    Yfilt2 = fftpack.ifftshift(Yfilt)
    yfilt = fftpack.ifft(Yfilt2)

    return np.real(yfilt)


def CenteredFFT(input, dt=1.0):
    N = len(input)

    # Generate frequency index
    if np.mod(N, 2) == 0:
        m = np.arange(-N/2, N/2-1+1)
    else:
        m = np.arange(-(N-1)/2, (N-1)/2+1)

    freq = m/(N*dt)
    X = fftpack.fft(input)
    X = fftpack.fftshift(X)

    return X, freq


def AliasFreq(f0, dt):

    fs = 1/dt

    lower = np.ceil(f0/fs - 0.5)
    upper = np.floor(f0/fs + 0.5)

    for n in [lower, upper, -lower, -upper]:
        fa = np.abs(f0 + n/dt)
        if fa <= 1/(2*dt):
            return fa

    raise ValueError('No integer found for aliasing')


def ConfChi2(alpha, dof):
    import numpy as np
    from scipy.stats import chi2

    return np.sort(dof/np.array(chi2.interval(1-alpha, dof)))


def SpectralDensity(input, dt=1, nsmooth=5, SubsetLength=None,
                    multitaper=False, breakpts=[]):
    """ Calculates spectral density for longest valid segment
        Direct translation of Tom's spectrum_band_avg.
        Always applies a Hann window.
        Input:
            input : variable whose spectral density you want
            dt : (optional) Δtime
            nsmooth : (optional) number of frequency bands to average over
            SubsetLength : (optional) Max length of data segment.
                            Spectra from multiple segments
                            are averaged.
            multitaper : Use multitaper method (default False)
            breakpts : List of breakpoints at which to change nsmooth

        Returns:
            S : estimated spectral density
            f : frequency bands
         conf : 95% confidence interval for 2*nsmooth dof
    """
    import scipy.signal as signal
    import dcpy.util
    import numpy as np
    import mtspec

    if SubsetLength is None:
        start, stop = FindLargestSegment(input)
        SubsetLength = stop - start
        start = [start]
        stop = [stop]
    else:
        start, stop = FindSegments(input)
        # Force subset length to be an integer
        SubsetLength = np.int(np.rint(SubsetLength))

    YY_raw = []
    for s0, s1 in zip(start, stop):
        SegmentLength = s1 - s0

        if SegmentLength < SubsetLength:
            continue

        for zz in range(s0, s1, SubsetLength+1):
            if zz+SubsetLength > s1:
                continue

            var = input[zz:zz+SubsetLength].copy()

            if np.any(np.isnan(var)):
                raise ValueError('Subset has NaNs!')

            N = len(var)
            T = N*dt
            window = signal.hann(N)
            # variance correction
            window /= np.sqrt(np.sum(window**2)/N)

            var -= var.mean()
            var = var * window

            if multitaper:
                Y, freq, conf, _, _ = mtspec.mtspec(
                    data=var, delta=dt, number_of_tapers=5,
                    time_bandwidth=4, statistics=True)
                Y = Y[freq > 0]
                freq = freq[freq > 0]
                conf = conf[freq > 0]
                YY_raw.append(Y)
            else:
                Y, freq = CenteredFFT(var, dt)
                Y = Y[freq > 0]
                freq = freq[freq > 0]
                confint = ConfChi2(0.05, 1)
                YY_raw.append(2*T/N**2 * Y * np.conj(Y))

    if YY_raw == []:
        raise ValueError('No subsets of specified length found.')

    # segment averaging
    if len(YY_raw) > 1:
        YY_raw = np.mean(np.abs(np.array(YY_raw)), axis=0)
    else:
        YY_raw = np.abs(YY_raw[0])

    # frequency band averaging
    if nsmooth is not None:
        if type(nsmooth) is not list:
            nsmooth = [nsmooth]
            breakpts = []
        else:
            nsmooth = nsmooth + [nsmooth[-1]]
            if breakpts == []:
                raise ValueError('SpectralDensity: ' +
                                 'Nsmooth is a list but breakpts is empty!')

            breakpts = [np.where(freq > bb)[0][0]
                        for bb in breakpts]

        breakpts.append(len(YY_raw))

        S = []
        f = []
        conf = []
        i1 = 0
        for idx, smth in enumerate(nsmooth):
            i0 = i1
            i1 = breakpts[idx]
            S.append(dcpy.util.MovingAverage(
                YY_raw[i0:i1], smth, decimate=True))
            f.append(dcpy.util.MovingAverage(
                freq[i0:i1], smth, decimate=True))

            confint = ConfChi2(0.05, 2*smth)
            if not multitaper:
                conf.append(np.array([confint[0]*S[idx],
                                      confint[1]*S[idx]]).T)

        S = np.concatenate(S)
        f = np.concatenate(f)
        conf = np.concatenate(conf)

    else:
        S = YY_raw
        f = freq
        if not multitaper:
            conf = np.array([confint[0]*S, confint[1]*S]).T

    mask = ~np.isnan(S)

    return S[mask], f[mask], conf[mask, :]


def Coherence(v1, v2, dt=1, nsmooth=5, **kwargs):
    from dcpy.util import MovingAverage

    mask = ~(np.isnan(v1) | np.isnan(v2))

    f, Cxy = signal.coherence(v1[mask], v2[mask], fs=1/dt, **kwargs)

    if np.mod(nsmooth, 2) < 1:
        nsmooth -= 1

    f = MovingAverage(f, nsmooth)
    Cxy = MovingAverage(Cxy, nsmooth)

    if nsmooth > 1:
        siglevel = np.sqrt(1 - (0.05)**(1/(nsmooth-1)))
    else:
        siglevel = 1

    return f, Cxy, siglevel


def BandPassButter(input, freqs, dt=1, order=1, **kwargs):

    b, a = signal.butter(N=order,
                         Wn=np.sort(freqs)*dt/(1/2),
                         btype='bandpass')

    return GappyFilter(input, b, a, num_discard=20)


def HighPassButter(input, freq, order=1):

    b, a = signal.butter(order, freq/(1/2), btype='high')

    return GappyFilter(input, b, a, 10)


def GappyFilter(input, b, a, num_discard=None):

    segstart, segend = FindSegments(input)
    out = np.empty(input.shape) * np.nan
    for index, start in np.ndenumerate(segstart):
        stop = segend[index]
        try:
            out[start:stop] = signal.filtfilt(b, a,
                                              input[start:stop],
                                              axis=0)
            if num_discard is not None:
                out[start:start+num_discard] = np.nan
                out[stop-num_discard:stop] = np.nan
        except ValueError:
            # segment is not long enough for filtfilt
            pass

    return out


def HighPassAndPlot(input, CutoffFreq, titlestr=None):

    import matplotlib.pyplot as plt
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
