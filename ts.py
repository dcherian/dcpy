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
                 SubsetLength=None, breakpts=[], multitaper=False,
                 preserve_area=False, scale=1, linearx=False, **kwargs):

    import matplotlib.pyplot as plt

    if ax is None:
            ax = plt.gca()

    var = np.array(var, ndmin=2)

    if var.shape[1] > var.shape[0]:
        var = var.T

    hdl = []
    for zz in range(var.shape[-1]):
        S, f, conf = SpectralDensity(var[:, zz]/(scale)**zz, dt,
                                     nsmooth, SubsetLength,
                                     breakpts=breakpts, multitaper=multitaper)

        if preserve_area:
            S = S*f
            conf = conf*f[:, np.newaxis]

        hdl.append(ax.plot(f, S, **kwargs)[0])
        if len(conf) > 2:
            ax.fill_between(f, conf[:, 0], conf[:, 1],
                            color=hdl[-1].get_color(), alpha=0.3)

    ax.set_yscale('log')
    if not linearx:
        ax.set_xscale('log')

    if len(hdl) == 1:
        hdl = hdl[0]

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

    input -= input.mean()

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
        Always applies a Hann window if not multitaper.
        Input:
            input : variable whose spectral density you want
            dt : (optional) Δtime
            nsmooth : (optional) number of frequency bands to average over
                      OR time-bandwidth product for multitaper.
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

            if multitaper:
                Y, freq, conf, _, _ = mtspec.mtspec(
                    data=var, delta=dt, time_bandwidth=nsmooth,
                    statistics=True, verbose=False, adaptive=True)
                Y = Y[freq > 0]
                freq = freq[freq > 0]
                conf = conf[freq > 0]
                YY_raw.append(Y)
            else:
                N = len(var)
                T = N * dt
                window = signal.hann(N)
                # variance correction
                window /= np.sqrt(np.sum(window**2)/N)

                Y, freq = CenteredFFT(var*window, dt)
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
    if nsmooth is not None and not multitaper:
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

    if np.any(np.isnan(v1) | np.isnan(v2)):
        raise ValueError('NaNs in times series provided to Coherence')

    y1, freq = CenteredFFT(v1, dt)
    y1 = y1[freq > 0]
    freq = freq[freq > 0]

    y2, freq = CenteredFFT(v2, dt)
    y2 = y2[freq > 0]
    freq = freq[freq > 0]

    P12 = MovingAverage(y1 * np.conj(y2), nsmooth)
    P11 = MovingAverage(y1 * np.conj(y1), nsmooth)
    P22 = MovingAverage(y2 * np.conj(y2), nsmooth)

    f = MovingAverage(freq, nsmooth)
    C = P12/np.sqrt(P11*P22)

    Cxy = np.abs(C)
    phase = np.angle(C)*180/np.pi
    if nsmooth > 1:
        siglevel = np.sqrt(1 - (0.05)**(1/(nsmooth-1)))
    else:
        siglevel = 1

    return f, Cxy, phase, siglevel


def MultiTaperCoherence(y0, y1, dt=1, tbp=4):
    from mtspec import mt_coherence

    # common defaults are time-bandwidth product tbp=4
    # ntapers = 2*tbp - 1 (jLab)

    if np.all(np.equal(y0, y1)):
        raise ValueError('Multitaper autocoherence doesn\'t work!')

    ntapers = 2*tbp - 1
    out = mt_coherence(1/dt, y0, y1, tbp=tbp, kspec=ntapers,
                       nf=np.int(len(y0)), p=0.95, iadapt=1,
                       freq=True, cohe=True, phase=True)

    if ntapers > 1:
        siglevel = np.sqrt(1 - (0.05)**(1/(ntapers-1)))
    else:
        siglevel = 1

    return out['freq'], out['cohe'], out['phase'], siglevel


def PlotCoherence(y0, y1, nsmooth=5, multitaper=False):

    import matplotlib.pyplot as plt

    if multitaper:
        f, Cxy, phase, siglevel = MultiTaperCoherence(y0, y1, nsmooth)
    else:
        f, Cxy, phase, siglevel = Coherence(y0, y1, nsmooth)

    plt.subplot(211)
    plt.plot(f, Cxy)
    plt.axhline(siglevel, color='gray', linestyle='--', zorder=-1)
    plt.title(str(sum(Cxy > siglevel)/len(Cxy)*100)
              + '% above 95% significance')
    plt.ylim([0, 1])

    plt.subplot(212)
    plt.plot(f, phase)


def BandPassButter(input, freqs, dt=1, order=1,
                   num_discard='auto', returnba=False):

    b, a = signal.butter(N=order,
                         Wn=np.sort(freqs)*dt/(1/2),
                         btype='bandpass')

    if returnba:
        return b, a
    else:
        return GappyFilter(input, b, a, num_discard=num_discard)


def ImpulseResponse(b, a, eps=1e-2):

    import matplotlib.pyplot as plt

    implen = EstimateImpulseResponseLength(b, a, eps=eps)
    ntime = implen*4

    x = np.arange(0, ntime)
    impulse = np.repeat(1, ntime)
    response = GappyFilter(impulse, b, a, num_discard=None)
    step = np.cumsum(response)

    plt.subplot(211)
    plt.plot(x, impulse, color='gray')
    plt.stem(x, response)
    plt.legend(['input', 'response'])
    plt.ylabel('Amplitude')
    plt.xlabel('n (samples)')
    plt.title('Response differences drops to ' + str(eps) + ' in '
              + str(implen) + ' samples.')
    plt.axvline(implen + int(ntime/2))
    plt.axvline(-implen + int(ntime/2))

    plt.subplot(212)
    plt.stem(x, step)
    plt.ylabel('Amplitude')
    plt.xlabel('n (samples)')
    plt.title('Step response')
    plt.axvline(implen)
    plt.subplots_adjust(hspace=0.5)
    plt.show()


def HighPassButter(input, freq, order=1):

    b, a = signal.butter(order, freq/(1/2), btype='high')

    return GappyFilter(input, b, a, 10)


def EstimateImpulseResponseLength(b, a, eps=1e-2):
    ''' From scipy filtfilt docs.
        Input:
             b, a : filter params
             eps  : How low must the signal drop to? (default 1e-2)
    '''

    z, p, k = signal.tf2zpk(b, a)
    r = np.max(np.abs(p))
    approx_impulse_len = int(np.ceil(np.log(eps)/np.log(r)))

    return approx_impulse_len


def GappyFilter(input, b, a, num_discard=None):

    if input.ndim == 1:
        input = np.reshape(input, (len(input), 1))

    out = np.empty(input.shape) * np.nan

    if num_discard == 'auto':
        num_discard = EstimateImpulseResponseLength(b, a, eps=1e-2)

    for ii in range(input.shape[1]):
        segstart, segend = FindSegments(input[:, ii])

        for index, start in np.ndenumerate(segstart):
            stop = segend[index]
            try:
                out[start:stop, ii] = \
                          signal.filtfilt(b, a,
                                          input[start:stop, ii],
                                          axis=0, method='gust',
                                          irlen=num_discard)
                if num_discard is not None:
                    out[start:start+num_discard, ii] = np.nan
                    out[stop-num_discard:stop, ii] = np.nan
            except ValueError:
                # segment is not long enough for filtfilt
                pass

    return out.squeeze()


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
