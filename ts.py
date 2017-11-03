import numpy as np
import scipy.signal as signal
import scipy.fftpack as fftpack
import matplotlib.pyplot as plt
import xarray as xr


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


def FindGaps(var):
    '''
      Finds and returns index ranges for gaps in the input time series.
      Input:
            var - input time series
      Output:
            start - starting indices of gap (NaN)
            stop  - ending indices of gap (NaN)
    '''

    NotNans = np.double(~np.isnan(var))
    edges = np.diff(NotNans)
    start = np.where(edges == -1)[0]
    stop = np.where(edges == 1)[0]

    if start.size == 0 and stop.size == 0:
        start = np.array([0])
        stop = np.array([len(var)-1])

    else:
        start = start + 1
        if np.isnan(var[0]):
            start = np.insert(start, 0, 0)

        if np.isnan(var[-1]):
            stop = np.append(stop, len(var)-1)

    return start, stop


def PlotSpectrum(var, ax=None, dt=1, nsmooth=5,
                 SubsetLength=None, breakpts=[], multitaper=False,
                 preserve_area=False, scale=1, linearx=False,
                 twoside=True, **kwargs):

    iscomplex = not np.all(np.isreal(var))
    if not iscomplex:
        twoside = False  # meaningless for real data

    if ax is None:
        ax = []
        if iscomplex and twoside is True:
            plt.figure(figsize=(8.5, 8.5/2.2))
            ax.append(plt.subplot(121))
            ax.append(plt.subplot(122))
            ax[0].set_title('CW (anti-cyclonic)')
            ax[1].set_title('CCW (cyclonic)')
        else:
            ax.append(plt.gca())
            ax.append(plt.gca())
            plt.gcf().set_size_inches(8.5, 8.5/1.617)

    var = np.array(var, ndmin=2)

    if type(ax) is not list:
        ax = [ax]

    if var.shape[1] > var.shape[0]:
        var = var.T

    hdl = []
    for zz in range(var.shape[-1]):
        if not iscomplex:
            S, f, conf = SpectralDensity(var[:, zz]/(scale)**(zz+1), dt,
                                         nsmooth, SubsetLength,
                                         breakpts=breakpts,
                                         multitaper=multitaper)

            if preserve_area:
                S = S*f
                conf = conf*f[:, np.newaxis]

            hdl.append(ax[0].plot(f, S, **kwargs)[0])
            if len(conf) > 2:
                ax[0].fill_between(f, conf[:, 0], conf[:, 1],
                                color=hdl[-1].get_color(), alpha=0.3)

        else:
            cw, ccw, f, conf_cw, conf_ccw = \
                    RotaryPSD(var[:, zz]/(scale)**(zz+1), dt,
                              multitaper=multitaper)
            hdl.append(ax[0].plot(f, cw, **kwargs)[0])
            if conf_cw != []:
                ax[0].fill_between(f, conf_cw[:, 0], conf_cw[:, 1],
                                   color=hdl[-1].get_color(), alpha=0.3)

            hdl.append(ax[1].plot(f, ccw, **kwargs)[0])
            if conf_ccw != []:
                ax[1].fill_between(f, conf_ccw[:, 0], conf_ccw[:, 1],
                                   color=hdl[-1].get_color(), alpha=0.3)

    for aa in ax:
        aa.set_yscale('log')
        if not linearx:
            aa.set_xscale('log')

        aa.set_xlabel('Freq')

    ax[0].set_ylabel('PSD')

    if not twoside:
        if iscomplex:
            ax[0].legend(['CW', 'CCW'])

        hdl = hdl[0]
        ax = ax[0]
    else:
        if not ax[0].xaxis_inverted():
            ax[0].invert_xaxis()

        ax[1].set_yticklabels([])
        plt.tight_layout()

    return hdl, ax


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

    if (np.sum(abs(X)**2)/N - np.sum(input**2))/np.sum(input**2) > 1e-3:
        raise ValueError('Parseval\'s theorem not satisfied!')

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


def TidalAliases(dt, kind='freq'):
    ''' Returns alias frequencies of tides as a dictionary.

        Input:
              dt = sampling time interval (days)

    '''

    # values from, and agree with, Schlax & Chelton (1994)
    TideAlias = dict()
    if kind == 'freq':
        TideAlias['M2'] = AliasFreq(1/(12.420601/24), dt)
        TideAlias['S2'] = AliasFreq(1/(12.0/24), dt)
        TideAlias['N2'] = AliasFreq(1/(12.658348/24), dt)
        TideAlias['K1'] = AliasFreq(1/(23.93447/24), dt)
        TideAlias['O1'] = AliasFreq(1/(25.819342/24), dt)
        TideAlias['P1'] = AliasFreq(1/(24.06589/24), dt)

    if kind == 'period':
        TideAlias['M2'] = 1/AliasFreq(1/(12.420601/24), dt)
        TideAlias['S2'] = 1/AliasFreq(1/(12.0/24), dt)
        TideAlias['N2'] = 1/AliasFreq(1/(12.658348/24), dt)
        TideAlias['K1'] = 1/AliasFreq(1/(23.93447/24), dt)
        TideAlias['O1'] = 1/AliasFreq(1/(25.819342/24), dt)
        TideAlias['P1'] = 1/AliasFreq(1/(24.06589/24), dt)

    return TideAlias


def ConfChi2(alpha, dof):
    import numpy as np
    from scipy.stats import chi2

    return np.sort(dof/np.array(chi2.interval(1-alpha, dof)))


def SpectralDensity(input, dt=1, nsmooth=5, SubsetLength=None,
                    multitaper=False, fillgaps=False, maxlen=None, breakpts=[]):
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
            fillgaps : (logical) Fill gaps < maxlen in input
            maxlen : Maximum length of gaps to fill (None for all gaps)

        Returns:
            S : estimated spectral density
            f : frequency bands
         conf : 95% confidence interval for 2*nsmooth dof
    """
    import dcpy.util
    import numpy as np
    import mtspec

    if fillgaps:
        input = FillGaps(input, maxlen=maxlen)

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

            var = signal.detrend(var)

            if multitaper:
                Y, freq, conf, _, _ = mtspec.mtspec(
                    data=var, delta=dt, time_bandwidth=nsmooth,
                    statistics=True, verbose=False, adaptive=True)
                Y = Y[freq > 0]
                conf = conf[freq > 0, :]
                freq = freq[freq > 0]
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
    from scipy.signal import detrend

    if np.any(np.isnan(v1) | np.isnan(v2)):
        raise ValueError('NaNs in times series provided to Coherence')

    window = signal.hann(len(v1))
    # variance correction
    window /= np.sqrt(np.sum(window**2)/len(v1))

    y1, freq = CenteredFFT(detrend(v1)*window, dt)
    y1 = y1[freq > 0]
    freq = freq[freq > 0]

    y2, freq = CenteredFFT(detrend(v2)*window, dt)
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


def MultiTaperCoherence(y0, y1, dt=1, tbp=5, ntapers=None):
    '''
        Call out to mt_coherence from mtspec.
        Phase is φ(y0) - φ(y1)
    '''

    from mtspec import mt_coherence
    from dcpy.util import calc95

    # common defaults are time-bandwidth product tbp=4
    # ntapers = 2*tbp - 1 (jLab)

    if np.all(np.equal(y0, y1)):
        raise ValueError('Multitaper autocoherence doesn\'t work!')

    if ntapers is None:
        ntapers = 2*tbp - 1

    from scipy.signal import detrend

    nf = np.int(len(y0)/2)+1
    out = mt_coherence(dt, detrend(y0), detrend(y1),
                       tbp=tbp, kspec=ntapers,
                       nf=nf, p=0.95, iadapt=1,
                       freq=True, cohe=True, phase=True,
                       cohe_ci=False, phase_ci=False)

    f = out['freq']
    cohe = out['cohe']
    phase = out['phase']

    # if ntapers > 1:
    #     siglevel = np.sqrt(1 - (0.05)**(1/(tbp/2*ntapers-1)))
    # else:
    #     siglevel = 1

    # monte-carlo significance level agrees with tbp/2*ntapers
    # being (degrees of freedom)/2
    niters = 1500
    y1 = np.random.randn(len(y0), niters)

    c = []
    for ii in range(niters):
        out = mt_coherence(dt, y0, y1[:, ii], tbp=tbp, kspec=ntapers,
                           nf=nf, p=0.95, iadapt=1,
                           freq=True, cohe=True)
        c.append(out['cohe'])

    siglevel = calc95(np.concatenate(c), 'onesided')

    return f, cohe, phase, siglevel


def RotaryPSD(y, dt=1, nsmooth=5, multitaper=False):

    """
    Inputs
    ------
        y : Time series
       dt : sampling period
    multitaper : Boolean
    nsmooth :
        amount of smoothing
        number of points if multitaper=False,
        tbp if multitaper=True

    Returns
    -------
    (cw, ccw, freq,conf_cw, conf_ccw)
    Clockwise, counter-clockwise, freq and confidence intervals.

    Confidence intervals for multitaper not implemented!
    """
    from scipy.signal import detrend
    from dcpy.util import MovingAverage

    N = len(y)

    if multitaper is True:
        import mtspec
        _, freq, xspec, X, _ = mtspec.mtspec(
                        data=detrend(np.real(y)), delta=dt,
                        time_bandwidth=nsmooth, optional_output=True,
                        statistics=False, verbose=False, adaptive=False)

        _, freq, yspec, Y, _ = mtspec.mtspec(
                        data=detrend(np.imag(y)), delta=dt,
                        time_bandwidth=nsmooth, optional_output=True,
                        statistics=False, verbose=False, adaptive=False)

        # for some reason, this normalization is needed :|
        X *= np.sqrt(N)
        Y *= np.sqrt(N)

    else:
        window = signal.hann(N)
        window /= np.sqrt(np.sum(window**2)/N)

        X, freq = CenteredFFT(detrend(np.real(y))*window, dt)
        Y, freq = CenteredFFT(detrend(np.imag(y))*window, dt)

    Gxx = dt/N * X * np.conjugate(X)
    Gyy = dt/N * Y * np.conjugate(Y)
    Qxy = dt/N * (np.real(X)*np.imag(Y) - np.imag(X)*np.real(Y))

    if multitaper:
        Gxx = np.mean(Gxx[0:len(freq)], axis=1)
        Gyy = np.mean(Gyy[0:len(freq)], axis=1)
        Qxy = np.mean(Qxy[0:len(freq)], axis=1)

    else:
        Gxx = MovingAverage(Gxx, nsmooth, decimate=True)
        Gyy = MovingAverage(Gyy, nsmooth, decimate=True)
        Qxy = MovingAverage(Qxy, nsmooth, decimate=True)
        freq = MovingAverage(freq, nsmooth, decimate=True)

    cw = 0.5 * (Gxx + Gyy - 2*Qxy)[freq > 0]
    ccw = 0.5 * (Gxx + Gyy + 2*Qxy)[freq > 0]
    freq = freq[freq > 0]

    if multitaper:
        conf_cw = []
        conf_ccw = []
    else:
        confint = np.array(ConfChi2(0.05, 2*nsmooth))[np.newaxis, :]
        conf_cw = confint * cw[:, np.newaxis]
        conf_ccw = confint * ccw[:, np.newaxis]

    return cw, ccw, freq, conf_cw, conf_ccw


def PlotCoherence(y0, y1, dt=1, nsmooth=5, multitaper=False, scale=1):

    import dcpy.plots

    if multitaper:
        f, Cxy, phase, siglevel = MultiTaperCoherence(y0, y1,
                                                      dt=dt,
                                                      tbp=nsmooth)
    else:
        f, Cxy, phase, siglevel = Coherence(y0, y1, dt=dt,
                                            nsmooth=nsmooth)

    plt.figure(figsize=(8, 9))
    ax1 = plt.subplot(311)
    PlotSpectrum(y0, ax=ax1, dt=dt, scale=scale,
                 nsmooth=nsmooth, multitaper=multitaper)
    PlotSpectrum(y1, ax=ax1, dt=dt, scale=scale,
                 nsmooth=nsmooth, multitaper=multitaper)
    ax1.invert_xaxis()

    plt.subplot(312, sharex=ax1)
    plt.plot(f, Cxy)
    dcpy.plots.liney(siglevel)
    plt.title('{0:.2f}'.format(sum(Cxy > siglevel)/len(Cxy)*100)
              + '% above 95% significance')
    plt.ylim([0, 1])

    plt.subplot(313, sharex=ax1)
    plt.plot(f, phase)
    plt.title('+ve = y0 leads y1')

    plt.tight_layout()


def BandPassButter(input, freqs, dt=1, order=1,
                   num_discard='auto', axis=-1, dim=None,
                   returnba=False):

    b, a = signal.butter(N=order,
                         Wn=np.sort(freqs)*dt/(1/2),
                         btype='bandpass')

    if returnba:
        return b, a
    else:
        if type(input) is xr.core.dataarray.DataArray:
            if dim is None:
                raise ValueError('Specify dim along which to band-pass')

            if type(dim) is list:
                dim = dim[0]

            x = input.copy()
            old_dims = x.dims
            idim = input.get_axis_num(dim)
            stackdims = x.dims[:idim] + x.dims[idim+1:]

            # xr.testing.assert_equal(x,
            #                         x.stack(newdim=stackdims)
            #                          .unstack('newdim')
            #                          .transpose(*list(x.dims)))
            if input.ndim > 2:
                # reshape to 2D array
                # 'dim' is now first index
                x = x.stack(newdim=stackdims)

            x.values = np.apply_along_axis(GappyFilter, 0,
                                           x.values,
                                           b, a,
                                           num_discard=num_discard)

            if input.ndim > 2:
                # unstack back to original shape and ordering
                x = x.unstack('newdim').transpose(*list(old_dims))

            return x
        else:
            return np.apply_along_axis(GappyFilter, axis, input,
                                  b, a, num_discard=num_discard)


def ImpulseResponse(b, a, eps=1e-2):

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


def oldGappyFilter(input, b, a, num_discard=None):

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


def GappyFilter(input, b, a, num_discard=None):

    out = np.empty(input.shape) * np.nan

    if num_discard == 'auto':
        num_discard = EstimateImpulseResponseLength(b, a, eps=1e-2)

    segstart, segend = FindSegments(input)

    for index, start in np.ndenumerate(segstart):
        stop = segend[index]
        try:
            out[start:stop] = signal.filtfilt(b, a,
                                              input[start:stop],
                                              axis=0, method='gust',
                                              irlen=num_discard)
            if num_discard is not None:
                out[start:start+num_discard] = np.nan
                out[stop-num_discard:stop] = np.nan
        except ValueError:
            # segment is not long enough for filtfilt
            pass

    return out.squeeze()


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


def FillGaps(y, x=None, maxlen=None):
    ''' Use linear interpolation to fill gaps < maxlen
        Input:
            y : value vector with gaps
            x : [optional] x (time) vector
            maxlen : max length of gaps to be filled
        Output:
            interpolated array
    '''

    import numpy as np
    import xarray as xr
    if isinstance(y, xr.core.dataarray.DataArray):
        isxarray = True
    else:
        isxarray = False

    if x is None:
        x = np.arange(len(y))

    if maxlen is None:
        yfill = y.copy()
        # fill all gaps
        valid = np.logical_not(np.isnan(y))
        if isxarray:
            yfill.values = np.interp(x, x[valid], y[valid])
        else:
            yfill = np.interp(x, x[valid], y[valid])
    else:
        yfill = y.copy()
        # fill only gaps < gaplen
        gapstart, gapstop = FindGaps(y)

        for g0, g1 in zip(gapstart, gapstop):
            if g0 == 0:
                continue

            if g1 == len(y)-1:
                continue

            glen = g1 - g0 + 1  # gap length
            if glen > maxlen:
                continue

            yfill[g0:g1+1] = np.interp(x[g0:g1+1],
                                       x[[g0-1, g1+1]],
                                       y[[g0-1, g1+1]])

    if isxarray:
        yfill.attrs['GapFilled'] = 'True'
        yfill.attrs['MaxGapLen'] = maxlen
        if 'numTimesGapFilled' in yfill.attrs:
            yfill.attrs['numTimesGapFilled'] += 1
        else:
            yfill.attrs['numTimesGapFilled'] = 1

    return yfill


def Spectrogram(var, window, shift, time=None, **kwargs):

    spec = []
    window = np.int(window)
    shift = np.int(shift)

    for ii in np.arange(0, len(var), shift):
        if ii > (len(var)-window):
            break

        S, f, _ = SpectralDensity(var[ii:ii+window], **kwargs)
        spec.append(S)

    spec = np.stack(spec)

    if time is None:
        time = np.arange(0, len(var), shift)
    else:
        time = time.copy()[::shift]

    return f, spec, time[:spec.shape[0]]


def PlotSpectrogram(time, spec, ax=None):

    if ax is None:
        ax = plt.gca()

    plt.contourf(time, f, np.log10(spec.T))
    plt.gca().set_yscale('log')
    plt.colorbar()
