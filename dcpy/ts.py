import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as fftpack
import scipy.signal as signal
from functools import partial
from matplotlib import ticker

import xarray as xr

from .plots import linex
from .util import calc95, one_over


def _is_datetime_like(da) -> bool:
    import numpy as np

    if np.issubdtype(da.dtype, np.datetime64) or np.issubdtype(
        da.dtype, np.timedelta64
    ):
        return True

    try:
        import cftime

        if isinstance(da.data[0], cftime.datetime):
            return True
    except ImportError:
        pass

    return False


def _process_time(time, cycles_per="s"):

    time = time.copy()
    dt = np.nanmedian(
        np.diff(time.values).astype(np.timedelta64) / np.timedelta64(1, cycles_per)
    )

    time = np.cumsum(time.copy().diff(dim=time.dims[0]) / np.timedelta64(1, cycles_per))

    return dt, time


def xfilter(
    x, flen=None, kind="hann", dim="time", decimate=False, min_values=None, **kwargs
):
    """flen and x.dim should have same units!"""

    from .util import smooth

    if not isinstance(x, xr.DataArray):
        raise ValueError("xfilter only works on DataArrays!")

    if flen is None or kind is None:
        return x

    # determine Δdim
    dt = np.diff(x[dim][0:2].values)

    if np.issubdtype(dt.dtype, np.timedelta64):
        dt = dt.astype("timedelta64[s]").astype("float32")

    if kind == "mean":
        N = np.int(np.floor(flen / dt))

        if N == 0:
            print("xfilter: filter length not long enough!")
            return x

        if min_values is None:
            min_periods = 1
        elif min_values < 1:
            min_periods = np.int(np.ceil(min_values * N))
        else:
            min_periods = min_values

        a = x.rolling(time=N, center=True, min_periods=min_periods).mean()

        if decimate:
            seldict = dict()
            seldict[dim] = slice(N - 1, len(a["time"]) - N + 1, N)
            a = a.isel(**seldict)

    elif kind == "bandpass":
        flen = np.array(flen.copy())
        if len(flen) == 1:
            raise ValueError("Bandpass filtering requires two frequencies!")

        a = BandPassButter(x.copy(), 1 / flen, dt, dim=dim)

    elif kind == "lowpass":
        a = x.copy(data=LowPassButter(x.copy(), 1 / (flen / dt)))

    elif kind == "highpass":
        a = x.copy(data=HighPassButter(x.copy(), 1 / (flen / dt)))

    else:
        a = x.copy()
        a.values = smooth(
            x.values, flen / dt, window=kind, axis=x.get_axis_num(dim), **kwargs
        )

    return a


def FindLargestSegment(input):

    start, stop = FindSegments(input)
    GapLength = stop - start + 1
    imax = np.argmax(GapLength)

    return start[imax], stop[imax]


def FindSegments(var):
    """
    Finds and return valid index ranges for the input time series.
    Input:
          var - input time series
    Output:
          start - starting indices of valid ranges
          stop  - ending indices of valid ranges
    """

    if var.dtype.kind == "b":
        NotNans = np.int32(var)
    else:
        NotNans = np.int32(~np.isnan(var))

    edges = np.diff(NotNans)
    start = np.where(edges == 1)[0]
    stop = np.where(edges == -1)[0]
    if start.size == 0 and stop.size == 0:
        start = np.array([0])
        stop = np.array([len(var) - 1])

    else:
        start = start + 1
        if NotNans[0]:
            start = np.insert(start, 0, 0)

        if NotNans[-1]:
            stop = np.append(stop, len(var) - 1)

    return start, stop


def FindGaps(var):
    """
    Finds and returns index ranges for gaps in the input time series.
    Input:
          var - input time series
    Output:
          start - starting indices of gap (NaN)
          stop  - ending indices of gap (NaN)
    """

    NotNans = np.double(~np.isnan(var))
    edges = np.diff(NotNans)
    start = np.where(edges == -1)[0]
    stop = np.where(edges == 1)[0]

    if start.size == 0 and stop.size == 0:
        start = np.array([0])
        stop = np.array([len(var) - 1])

    else:
        start = start + 1
        if np.isnan(var[0]):
            start = np.insert(start, 0, 0)

        if np.isnan(var[-1]):
            stop = np.append(stop, len(var) - 1)

    return start, stop


def PlotSpectrum(
    var,
    ax=None,
    dt=1,
    nsmooth=5,
    SubsetLength=None,
    breakpts=[],
    multitaper=True,
    preserve_area=False,
    scale=1,
    linearx=False,
    axis=-1,
    twoside=False,
    decimate=True,
    mark_freqs=[],
    cycles_per="D",
    **kwargs,
):
    """
    Parameters
    ----------

    var: xr.DataArray, numpy.ndarray
    ax: matplotlib.Axes
    dt: spacing
    nsmooth: int
        number of points to smooth over using running average (see decima)
    SubsetLength: int,
        Split up time-series into segments of this length and then average
    multitaper: bool
        Calculate multitaper spectrum [requires mtspec]
    preserve_area: bool
        Plot area or variance preserving form (plots PSD x frequency)
    scale: float32
        scale plotted spectra by this amount
    linearx: bool
        Use linear freq scale
    axis: int
        Axis along which to calculate spectrum
    decimate: bool
        Decimate after using running average fopr smoothing
    mark_freqs: list
        Draw vertical lines at frequencies
    twoside: bool
        If provided with complex time series, plot either twosided spectra.
    cycles_per: str
        Convert datetime spacing to cycles_per_units.
    """

    iscomplex = not np.all(np.isreal(var))
    if not iscomplex:
        twoside = False  # meaningless for real data

    if isinstance(var, xr.DataArray):
        name = " " + xr.plot.utils.label_from_attrs(var)
    else:
        name = " "

    if ax is None:
        ax = []
        if iscomplex and twoside is True:
            f, ax = plt.subplots(
                1, 2, figsize=(8.5, 8.5 / 2.2), sharey=True, constrained_layout=True
            )
            ax[0].set_title("CW (anti-cyclonic)" + name)
            ax[1].set_title("CCW (cyclonic)" + name)
        else:
            f, aa = plt.subplots(constrained_layout=True)
            aa.set_title(name)
            ax.append(aa)
            ax.append(aa)
            plt.gcf().set_size_inches(8.5, 8.5 / 1.617)
    else:
        if twoside is False and not hasattr(ax, "__iter__"):
            ax.set_title(name)
            ax = [ax, ax]
        elif iscomplex and twoside is True:
            assert len(ax) == 2
            ax[0].set_title("CW (anti-cyclonic)" + name)
            ax[1].set_title("CCW (cyclonic)" + name)

    processed_time = False
    if isinstance(var, xr.DataArray) and var.ndim == 1:
        maybe_time = var[var.dims[0]]
        if _is_datetime_like(maybe_time):
            dt, t = _process_time(maybe_time, cycles_per)
            processed_time = True

    if isinstance(var, xr.DataArray):
        var = var.dropna(var.dims[0])

    if var.ndim == 1:
        var = np.array(var, ndmin=2)
        if var.shape[0] == 1:
            var = var.transpose()
            axis = -1

    # if var.shape[axis] == 1:
    #     var = var.transpose()
    #     if var.shape[-1] == 1:
    #         axis = -1

    if axis == 0:
        var = var.transpose()
        axis = -1

    if not hasattr(ax, "__iter__"):
        ax = [ax]

    hdl = []
    for zz in range(var.shape[1]):
        if not iscomplex:
            S, f, conf = SpectralDensity(
                var[:, zz] / (scale) ** (zz + 1),
                dt,
                nsmooth,
                SubsetLength,
                breakpts=breakpts,
                multitaper=multitaper,
                decimate=decimate,
            )

            if preserve_area:
                S = S * f
                conf = conf * f[:, np.newaxis]

            hdl.append(ax[0].plot(f, S, **kwargs)[0])
            if len(conf) > 2:
                ax[0].fill_between(
                    f, conf[:, 0], conf[:, 1], color=hdl[-1].get_color(), alpha=0.3
                )

        else:
            cw, ccw, f, conf_cw, conf_ccw = RotaryPSD(
                var[:, zz] / (scale) ** (zz + 1),
                dt,
                nsmooth=nsmooth,
                multitaper=multitaper,
            )

            if preserve_area:
                cw = cw * f
                ccw = ccw * f
                if len(conf_cw) > 0:
                    conf_ccw = conf_ccw * f[:, np.newaxis]
                if len(conf_ccw) > 0:
                    conf_cw = conf_cw * f[:, np.newaxis]

            hdl.append(ax[0].plot(f, cw, **kwargs)[0])

            if len(conf_cw) > 0:
                ax[0].fill_between(
                    f,
                    conf_cw[:, 0],
                    conf_cw[:, 1],
                    color=hdl[-1].get_color(),
                    alpha=0.3,
                )

            hdl.append(ax[1].plot(f, ccw, **kwargs)[0])
            if len(conf_ccw) > 0:
                ax[1].fill_between(
                    f,
                    conf_ccw[:, 0],
                    conf_ccw[:, 1],
                    color=hdl[-1].get_color(),
                    alpha=0.3,
                )

    for aa in ax:
        aa.set_yscale("log")
        if not linearx:
            aa.set_xscale("log")

        aa2 = aa.secondary_xaxis("top", functions=(one_over, one_over))

        if not processed_time:
            aa.set_xlabel("Wavelength/2π")
            aa2.set_xlabel("Wavenumber/2π")
        else:
            aa.set_xlabel("Frequency " + "[cp" + cycles_per.lower() + "]")
            aa2.set_xlabel("Period " + "[" + cycles_per.lower() + "]")

    if preserve_area:
        ax[0].set_ylabel("Freq x PSD")
        ax[0].set_yscale("linear")
    else:
        ax[0].set_ylabel("PSD")

    [aa.autoscale(enable=True, tight=True) for aa in ax]

    if not twoside:
        if iscomplex:
            ax[0].legend(["CW", "CCW"])

        ax = ax[0]
    else:
        if not ax[0].xaxis_inverted():
            ax[0].invert_xaxis()

        if preserve_area:
            ax[1].set_yscale("linear")

    if mark_freqs:
        linex(mark_freqs, ax=ax)

    return hdl, ax


def synthetic(N, dt, α, β):
    """
    Generate time series with spectrum S = α ω^β

    Input:
        N : number of points
        dt : time interval
        α, β : spectrum parameters

    Output:
        yfilt : time series with desired spectral shape

    Ack:
       Copied from Tom Farrar's synthetic_timeseries_known_spectrum
    """

    from numpy import sqrt

    y = np.random.randn(N)

    [Y, freq] = CenteredFFT(y, dt)

    Yfilt = sqrt(α) * sqrt(1.0 / (2 * dt)) * (np.abs(freq) ** (β / 2)) * Y

    ff = np.where(freq == 0)
    Yfilt[ff] = 0
    Yfilt2 = fftpack.ifftshift(Yfilt)
    yfilt = fftpack.ifft(Yfilt2)

    return np.real(yfilt)


def CenteredFFT(input, dt=1.0, axis=-1):
    N = len(input)

    # Generate frequency index
    if np.mod(N, 2) == 0:
        m = np.arange(-N / 2, N / 2 - 1 + 1)
    else:
        m = np.arange(-(N - 1) / 2, (N - 1) / 2 + 1)

    input = signal.detrend(input.copy())

    freq = m / (N * dt)

    X = fftpack.fft(input, axis=axis)
    X = fftpack.fftshift(X, axes=axis)

    if (np.sum(abs(X) ** 2) / N - np.sum(input ** 2)) / np.sum(input ** 2) > 1e-3:
        raise ValueError("Parseval's theorem not satisfied!")

    return X, freq


def AliasFreq(f0, dt):

    fs = 1 / dt

    lower = np.ceil(f0 / fs - 0.5)
    upper = np.floor(f0 / fs + 0.5)

    for n in [lower, upper, -lower, -upper]:
        fa = np.abs(f0 + n / dt)
        if fa <= 1 / (2 * dt):
            return fa

    raise ValueError("No integer found for aliasing")


def TidalAliases(dt, kind="freq"):
    """Returns alias frequencies of tides as a dictionary.

    Input:
          dt = sampling time interval (days)

    """

    # values from, and agree with, Schlax & Chelton (1994)
    TideAlias = dict()
    if kind == "freq":
        TideAlias["M2"] = AliasFreq(1 / (12.420601 / 24), dt)
        TideAlias["S2"] = AliasFreq(1 / (12.0 / 24), dt)
        TideAlias["N2"] = AliasFreq(1 / (12.658348 / 24), dt)
        TideAlias["K1"] = AliasFreq(1 / (23.93447 / 24), dt)
        TideAlias["O1"] = AliasFreq(1 / (25.819342 / 24), dt)
        TideAlias["P1"] = AliasFreq(1 / (24.06589 / 24), dt)

    if kind == "period":
        TideAlias["M2"] = 1 / AliasFreq(1 / (12.420601 / 24), dt)
        TideAlias["S2"] = 1 / AliasFreq(1 / (12.0 / 24), dt)
        TideAlias["N2"] = 1 / AliasFreq(1 / (12.658348 / 24), dt)
        TideAlias["K1"] = 1 / AliasFreq(1 / (23.93447 / 24), dt)
        TideAlias["O1"] = 1 / AliasFreq(1 / (25.819342 / 24), dt)
        TideAlias["P1"] = 1 / AliasFreq(1 / (24.06589 / 24), dt)

    return TideAlias


def ConfChi2(alpha, dof):
    import numpy as np
    from scipy.stats import chi2

    return np.sort(dof / np.array(chi2.interval(1 - alpha, dof)))


def SpectralDensity(
    input,
    dt=1,
    nsmooth=5,
    SubsetLength=None,
    multitaper=False,
    fillgaps=False,
    maxlen=None,
    breakpts=[],
    decimate=True,
):
    """Calculates spectral density for longest valid segment
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
        f : frequency bands (1/period)
            [sin(2π/10) will have a peak at 1/10]
     conf : 95% confidence interval for 2*nsmooth dof
    """
    import dcpy.util
    import numpy as np
    import mtspec

    if len(input) == 0:
        raise ValueError("0 length input!")

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

        for zz in range(s0, s1, SubsetLength + 1):
            if zz + SubsetLength > s1:
                continue

            var = input[zz : zz + SubsetLength - 1].copy()

            if np.any(np.isnan(var)):
                raise ValueError("Subset has NaNs!")

            var = signal.detrend(var)

            if multitaper:
                Y, freq, conf, _, _ = mtspec.mtspec(
                    data=var,
                    delta=dt,
                    time_bandwidth=nsmooth,
                    statistics=True,
                    verbose=False,
                    adaptive=True,
                )
                Y = Y[freq > 0]
                conf = conf[freq > 0, :]
                freq = freq[freq > 0]
                YY_raw.append(Y)
            else:
                N = len(var)
                T = N * dt
                window = signal.hann(N)
                # variance correction
                window /= np.sqrt(np.sum(window ** 2) / N)

                try:
                    Y, freq = CenteredFFT(var * window, dt)
                except ValueError:
                    Y, freq = CenteredFFT(var * np.atleast_2d(window).T, dt)
                Y = Y[freq > 0]
                freq = freq[freq > 0]
                confint = ConfChi2(0.05, 1)
                YY_raw.append(2 * T / N ** 2 * Y * np.conj(Y))

    if YY_raw == []:
        raise ValueError("No subsets of specified length found.")

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
                raise ValueError(
                    "SpectralDensity: " + "Nsmooth is a list but breakpts is empty!"
                )

            breakpts = [np.where(freq > bb)[0][0] for bb in breakpts]

        breakpts.append(len(YY_raw))

        S = []
        f = []
        conf = []
        i1 = 0
        for idx, smth in enumerate(nsmooth):
            i0 = i1
            i1 = breakpts[idx]
            S.append(dcpy.util.MovingAverage(YY_raw[i0:i1], smth, decimate=decimate))
            f.append(dcpy.util.MovingAverage(freq[i0:i1], smth, decimate=decimate))

            confint = ConfChi2(0.05, 2 * smth)
            conf.append(np.array([confint[0] * S[idx], confint[1] * S[idx]]).T)

        S = np.concatenate(S)
        f = np.concatenate(f)
        conf = np.concatenate(conf)

    else:
        S = YY_raw
        f = freq
        if not multitaper:
            conf = np.array([confint[0] * S, confint[1] * S]).T

    mask = ~np.isnan(S)

    return S[mask], f[mask], conf[mask, :]


def Coherence(v1, v2, dt=1, nsmooth=5, decimate=True, **kwargs):
    from scipy.signal import detrend

    from dcpy.util import MovingAverage

    if np.any(np.isnan(v1) | np.isnan(v2)):
        raise ValueError("NaNs in times series provided to Coherence")

    window = signal.hann(len(v1))
    # variance correction
    window /= np.sqrt(np.sum(window ** 2) / len(v1))

    y1, freq = CenteredFFT(detrend(v1) * window, dt)
    y1 = y1[freq > 0]
    freq = freq[freq > 0]

    y2, freq = CenteredFFT(detrend(v2) * window, dt)
    y2 = y2[freq > 0]
    freq = freq[freq > 0]

    P12 = MovingAverage(y1 * np.conj(y2), nsmooth, decimate=decimate)
    P11 = MovingAverage(y1 * np.conj(y1), nsmooth, decimate=decimate)
    P22 = MovingAverage(y2 * np.conj(y2), nsmooth, decimate=decimate)

    f = MovingAverage(freq, nsmooth, decimate=decimate)
    C = P12 / np.sqrt(P11 * P22)

    Cxy = C
    phase = np.angle(C, deg=True)
    if nsmooth > 1:
        siglevel = 1 - (0.05) ** (1 / (nsmooth - 1))
    else:
        siglevel = 1

    # C2std = 1.414 * (1 - Cxy**2) / np.abs(Cxy) / np.sqrt(nsmooth)

    # niter = 1000
    # white = np.random.randn(len(v1), niter)
    # yw, freq = CenteredFFT(detrend(white, 1) * window[:, np.newaxis], dt, axis=0)
    # yw = yw[freq > 0, :]
    # Pw2 = MovingAverage(yw * np.conj(y2[:, np.newaxis]),
    #                     nsmooth, decimate=decimate)
    # Pww = MovingAverage(yw * np.conj(yw), nsmooth, decimate=decimate)

    # Cxy_white = np.abs(Pw2 / np.sqrt(Pww * P22[:, np.newaxis]))
    # siglevel = calc95(Cxy_white.ravel(), 'onesided')

    return f, Cxy, phase, siglevel


def MultiTaperCoherence(y0, y1, dt=1, tbp=5, ntapers=None):
    """
    Call out to mt_coherence from mtspec.
    Phase is φ(y0) - φ(y1)
    """

    from mtspec import mt_coherence

    from dcpy.util import calc95

    # common defaults are time-bandwidth product tbp=4
    # ntapers = 2*tbp - 1 (jLab)

    if np.all(np.equal(y0, y1)):
        raise ValueError("Multitaper autocoherence doesn't work!")

    if ntapers is None:
        ntapers = 2 * tbp - 1

    from scipy.signal import detrend

    nf = np.int(len(y0) / 2) + 1
    out = mt_coherence(
        dt,
        detrend(y0),
        detrend(y1),
        tbp=tbp,
        kspec=ntapers,
        nf=nf,
        p=0.95,
        iadapt=1,
        freq=True,
        cohe=True,
        phase=True,
        cohe_ci=False,
        phase_ci=False,
    )

    f = out["freq"]
    cohe = out["cohe"] ** 2
    phase = out["phase"]

    if ntapers > 1:
        siglevel = 1 - (0.05) ** (1 / (tbp / 2 * ntapers - 1))
    else:
        siglevel = 1

    # monte-carlo significance level agrees with tbp/2*ntapers
    # being (degrees of freedom)/2
    # niters = 1500
    # y1 = np.random.randn(len(y0), niters)

    # c = []
    # for ii in range(niters):
    #     out = mt_coherence(dt, y0, y1[:, ii], tbp=tbp, kspec=ntapers,
    #                        nf=nf, p=0.95, iadapt=1,
    #                        freq=True, cohe=True)
    #     c.append(out['cohe'])

    # siglevel = calc95(np.concatenate(c), 'onesided')

    return f, cohe, phase, siglevel


def RotaryPSD(y, dt=1, nsmooth=5, multitaper=False, decimate=True):
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
            data=detrend(np.real(y)),
            delta=dt,
            time_bandwidth=nsmooth,
            optional_output=True,
            statistics=False,
            verbose=False,
            adaptive=False,
        )

        _, freq, yspec, Y, _ = mtspec.mtspec(
            data=detrend(np.imag(y)),
            delta=dt,
            time_bandwidth=nsmooth,
            optional_output=True,
            statistics=False,
            verbose=False,
            adaptive=False,
        )

        # for some reason, this normalization is needed :|
        X *= np.sqrt(N)
        Y *= np.sqrt(N)

    else:
        window = signal.hann(N)
        window /= np.sqrt(np.sum(window ** 2) / N)

        X, freq = CenteredFFT(detrend(np.real(y)) * window, dt)
        Y, freq = CenteredFFT(detrend(np.imag(y)) * window, dt)

    Gxx = dt / N * X * np.conjugate(X)
    Gyy = dt / N * Y * np.conjugate(Y)
    Qxy = -dt / N * (np.real(X) * np.imag(Y) - np.imag(X) * np.real(Y))

    if multitaper:
        Gxx = np.mean(Gxx[0 : len(freq)], axis=1)
        Gyy = np.mean(Gyy[0 : len(freq)], axis=1)
        Qxy = np.mean(Qxy[0 : len(freq)], axis=1)

    else:
        Gxx = MovingAverage(Gxx, nsmooth, decimate=decimate)
        Gyy = MovingAverage(Gyy, nsmooth, decimate=decimate)
        Qxy = MovingAverage(Qxy, nsmooth, decimate=decimate)
        if decimate is True:
            freq = MovingAverage(freq, nsmooth, decimate=True)

    cw = 0.5 * (Gxx + Gyy - 2 * Qxy)[freq > 0]
    ccw = 0.5 * (Gxx + Gyy + 2 * Qxy)[freq > 0]
    freq = freq[freq > 0]

    if multitaper:
        conf_cw = []
        conf_ccw = []
    else:
        confint = np.array(ConfChi2(0.05, 2 * nsmooth))[np.newaxis, :]
        conf_cw = confint * cw[:, np.newaxis]
        conf_ccw = confint * ccw[:, np.newaxis]

    return (np.real(cw), np.real(ccw), freq, np.real(conf_cw), np.real(conf_ccw))


def PlotCoherence(y0, y1, dt=1, nsmooth=5, multitaper=False, scale=1, decimate=False):

    import dcpy.plots

    if multitaper:
        f, Cxy, phase, siglevel = MultiTaperCoherence(y0, y1, dt=dt, tbp=nsmooth)
    else:
        f, Cxy, phase, siglevel = Coherence(
            y0, y1, dt=dt, nsmooth=nsmooth, decimate=decimate
        )

    fig, ax = plt.subplots(3, 1, sharex=True, constrained_layout=True)
    fig.set_size_inches((8, 9))
    PlotSpectrum(
        y0,
        ax=ax[0],
        dt=dt,
        scale=scale,
        nsmooth=nsmooth,
        multitaper=multitaper,
        decimate=decimate,
    )
    PlotSpectrum(
        y1,
        ax=ax[0],
        dt=dt,
        scale=scale,
        nsmooth=nsmooth,
        multitaper=multitaper,
        decimate=decimate,
    )

    ax[1].plot(f, Cxy)
    dcpy.plots.liney(siglevel, ax=ax[1])
    ax[1].set_title(
        f"{sum(Cxy > siglevel) / len(Cxy) * 100:.2f}% above 95% significance"
    )
    ax[1].set_ylim([0, 1])
    ax[1].set_ylabel("Squared Coherence")

    ax[2].plot(f, phase)
    ax[2].set_ylabel("Coherence phase")
    ax[2].set_title("+ve = y0 leads y1")

    return ax


def BandPassButter(
    input,
    freqs,
    dt=1,
    order=1,
    num_discard="auto",
    axis=-1,
    dim=None,
    returnba=False,
    debug=False,
):

    b, a = signal.butter(N=order, Wn=np.sort(freqs) * dt / (1 / 2), btype="bandpass")

    if returnba:
        return b, a
    else:
        if type(input) is xr.core.dataarray.DataArray:
            if len(input.dims) > 1 and dim is None:
                raise ValueError("Specify dim along which to band-pass")
            else:
                dim = input.dims[0]

            if type(dim) is list:
                dim = dim[0]

            x = input.copy()
            old_dims = x.dims
            idim = input.get_axis_num(dim)
            stackdims = x.dims[:idim] + x.dims[idim + 1 :]

            # xr.testing.assert_equal(x,
            #                         x.stack(newdim=stackdims)
            #                          .unstack('newdim')
            #                          .transpose(*list(x.dims)))
            if input.ndim > 2:
                # reshape to 2D array
                # 'dim' is now first index
                x = x.stack(newdim=stackdims)

            newdims = x.dims
            if newdims[0] != dim:
                x = x.transpose()

            x.values = np.apply_along_axis(
                GappyFilter, 0, x.values, b, a, num_discard=num_discard
            )

            if input.ndim > 2:
                # unstack back to original shape and ordering
                bp = x.unstack("newdim").transpose(*list(old_dims))
            else:
                bp = x

        else:
            bp = np.apply_along_axis(
                GappyFilter, axis, input, b, a, num_discard=num_discard
            )

        if debug is True:
            PlotSpectrum(input)
            PlotSpectrum(bp, ax=plt.gca())
            linex(freqs)

        return bp


def ImpulseResponse(b, a, eps=1e-2):

    implen = EstimateImpulseResponseLength(b, a, eps=eps)
    ntime = implen * 4

    x = np.arange(0, ntime)
    impulse = np.repeat(1, ntime)
    response = GappyFilter(impulse, b, a, num_discard=None)
    step = np.cumsum(response)

    plt.subplot(211)
    plt.plot(x, impulse, color="gray")
    plt.stem(x, response)
    plt.legend(["input", "response"])
    plt.ylabel("Amplitude")
    plt.xlabel("n (samples)")
    plt.title(
        "Response differences drops to " + str(eps) + " in " + str(implen) + " samples."
    )
    plt.axvline(implen + int(ntime / 2))
    plt.axvline(-implen + int(ntime / 2))

    plt.subplot(212)
    plt.stem(x, step)
    plt.ylabel("Amplitude")
    plt.xlabel("n (samples)")
    plt.title("Step response")
    plt.axvline(implen)
    plt.subplots_adjust(hspace=0.5)
    plt.show()


def LowPassButter(input, freq, order=1):

    b, a = signal.butter(order, freq / (1 / 2), btype="low")

    return GappyFilter(input, b, a)


def HighPassButter(input, freq, order=1):

    b, a = signal.butter(order, freq / (1 / 2), btype="high")

    return GappyFilter(input, b, a)


def EstimateImpulseResponseLength(b, a, eps=1e-2):
    """From scipy filtfilt docs.
    Input:
         b, a : filter params
         eps  : How low must the signal drop to? (default 1e-2)
    """

    z, p, k = signal.tf2zpk(b, a)
    r = np.max(np.abs(p))
    approx_impulse_len = int(np.ceil(np.log(eps) / np.log(r)))

    return approx_impulse_len


def oldGappyFilter(input, b, a, num_discard=None):

    if input.ndim == 1:
        input = np.reshape(input, (len(input), 1))

    out = np.empty(input.shape) * np.nan

    if num_discard == "auto":
        num_discard = EstimateImpulseResponseLength(b, a, eps=1e-2)

    for ii in range(input.shape[1]):
        segstart, segend = FindSegments(input[:, ii])

        for index, start in np.ndenumerate(segstart):
            stop = segend[index]
            try:
                out[start:stop, ii] = signal.filtfilt(
                    b,
                    a,
                    input[start:stop, ii],
                    axis=0,
                    method="gust",
                    irlen=num_discard,
                )
                if num_discard is not None:
                    out[start : start + num_discard, ii] = np.nan
                    out[stop - num_discard : stop, ii] = np.nan
            except ValueError:
                # segment is not long enough for filtfilt
                pass

    return out.squeeze()


def GappyFilter(input, b, a, num_discard="auto"):

    out = np.empty(input.shape) * np.nan

    if num_discard == "auto":
        num_discard = EstimateImpulseResponseLength(b, a)

    segstart, segend = FindSegments(input)

    for index, start in np.ndenumerate(segstart):
        stop = segend[index]
        try:
            out[start:stop] = signal.filtfilt(
                b, a, input[start:stop], axis=0, method="gust", irlen=num_discard
            )
            if num_discard is not None:
                out[start : start + num_discard] = np.nan
                out[stop - num_discard : stop] = np.nan
        except ValueError:
            # segment is not long enough for filtfilt
            pass

    return out.squeeze()


def HighPassAndPlot(input, CutoffFreq, titlestr=None, **kwargs):

    start, stop = FindLargestSegment(input)
    filtered = HighPassButter(input, CutoffFreq)

    ax = []
    plt.figure()
    ax.append(plt.subplot(4, 1, 1))
    PlotSpectrum(input, ax=ax[0], **kwargs)
    PlotSpectrum(filtered, ax=ax[0], **kwargs)
    ax[0].legend(["input", "filtered"])
    ax[0].axvline(CutoffFreq, color="gray", zorder=-20)
    ax[0].set_title(titlestr)

    ax.append(plt.subplot(4, 1, 2))
    ax[1].plot(input)

    ax.append(plt.subplot(4, 1, 3, sharex=ax[1]))
    ax[2].plot(filtered)

    period = np.int(np.round(1 / CutoffFreq))
    ax.append(plt.subplot(4, 1, 4, sharex=ax[1]))
    PlotSpectrogram(input, nfft=5 * period, shift=2 * period)
    ax[3].liney(CutoffFreq)

    return filtered


def apply_along_dim_1d(invar, dim, func, args=(), **kwargs):

    x = invar.copy()
    idim = invar.get_axis_num(dim)
    stackdims = x.dims[:idim] + x.dims[idim + 1 :]

    if invar.ndim > 2:
        # reshape to 2D
        # 'dim' is now first index
        x = x.stack(newdim=stackdims)

    x.values = np.apply_along_axis(func, 0, x.values, args, **kwargs)

    if invar.ndim > 2:
        # unstack back to original shape and ordering
        x = x.unstack("newdim").transpose(*list(invar.dims))

    return x


def FillGaps(y, x=None, maxlen=None):
    """TODO: use pandas.fillna
    Use linear interpolation to fill gaps < maxlen
    Input:
        y : value vector with gaps
        x : [optional] x (time) vector
        maxlen : max length of gaps to be filled
    Output:
        interpolated array
    """

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

            if g1 == len(y) - 1:
                continue

            glen = g1 - g0 + 1  # gap length
            if glen > maxlen:
                continue

            yfill[g0 : g1 + 1] = np.interp(
                x[g0 : g1 + 1], x[[g0 - 1, g1 + 1]], y[[g0 - 1, g1 + 1]]
            )

    if isxarray:
        yfill.attrs["GapFilled"] = "True"
        yfill.attrs["MaxGapLen"] = maxlen
        if "numTimesGapFilled" in yfill.attrs:
            yfill.attrs["numTimesGapFilled"] += 1
        else:
            yfill.attrs["numTimesGapFilled"] = 1

    return yfill


def Spectrogram(var, nfft, shift, time=None, dim=None, **kwargs):

    if time is None and dim is not None and isinstance(var, xr.DataArray):
        time = var[dim]
    elif time is None and isinstance(var, xr.DataArray):
        dim = var[var.dims[0]]

    iscomplex = np.any(np.iscomplex(var))

    if iscomplex:
        spec_cw = []
        spec_ccw = []
    else:
        spec = []

    nfft = np.int(nfft)
    shift = np.int(shift)
    start = np.int(np.floor(nfft / 2))
    nb2 = np.int(np.floor(nfft / 2))

    for ii in np.arange(start, len(var), shift):
        if ii + nb2 > len(var):
            break

        if iscomplex:
            cw, ccw, f, _, _ = RotaryPSD(var[ii - nb2 : ii + nb2], **kwargs)
            spec_cw.append(cw)
            spec_ccw.append(ccw)
        else:
            S, f, _ = SpectralDensity(var[ii - nb2 : ii + nb2], **kwargs)
            spec.append(S)

    if time is None:
        time = np.arange(start, ii, shift)
    else:
        time = time.copy()[np.arange(start, ii, shift)]

    if iscomplex:
        spec = xr.Dataset()
        spec["cw"] = xr.DataArray(
            np.stack(spec_cw), dims=["time", "freq"], coords=[time, f], name="CW PSD"
        )

        spec["ccw"] = xr.DataArray(
            np.stack(spec_ccw), dims=["time", "freq"], coords=[time, f], name="CCW PSD"
        )

    else:
        spec = xr.DataArray(
            np.stack(spec), dims=["time", "freq"], coords=[time, f], name="PSD"
        )

    return spec


def PlotSpectrogram(da, nfft, shift, multitaper=False, ax=None, **kwargs):

    iscomplex = np.any(np.iscomplex(da))

    if ax is None:
        if iscomplex:
            f, ax = plt.subplots(3, 1, sharex=True, constrained_layout=True)
        else:
            f, ax = plt.subplots(2, 1, sharex=True, constrained_layout=True)

    spec = Spectrogram(
        da, nfft, shift, dim=da.dims[0], decimate=False, multitaper=multitaper, **kwargs
    )

    plot_kwargs = dict(
        x=da.dims[0], yscale="log", cmap=svc.cm.blue_orange_div, robust=True
    )
    mtitle = " [mutitaper]" if multitaper else " [freq. smoothed]"

    if iscomplex:
        np.real(da).plot.line(x=da.dims[0], ax=ax[0])
        np.imag(da).plot.line(x=da.dims[0], ax=ax[0])
        ax[0].legend(("real", "imag"))

        hdl = (spec.cw * spec.freq).plot.contourf(ax=ax[1], levels=25, **plot_kwargs)

        ax[1].set_title("Variance-preserving CW spectrogram" + mtitle)
        (spec.ccw * spec.freq).plot.contourf(ax=ax[2], levels=hdl.levels, **plot_kwargs)
        ax[2].set_title("Variance-preserving CCW spectrogram" + mtitle)

    else:
        da.plot.line(x=da.dims[0], ax=ax[0])

        (spec * spec.freq).plot.contourf(ax=ax[1], levels=25, **plot_kwargs)
        ax[1].set_title("Variance-preserving spectrogram" + mtitle)

    [aa.set_xlabel("") for aa in ax[:-1]]

    return ax


def wavelet(var, dt=1):
    import wavelets

    wave, period, scale, coi = wavelets.wavelet(var - np.nanmean(var), dt=dt, pad=1)

    if isinstance(var, xr.DataArray):
        if var.ndim > 1:
            raise ValueError("Only 1-D data supported!")

        dim = var.dims[0]

    wave = xr.DataArray(
        wave, dims=["period", dim], coords={"period": period, dim: var[dim]}
    )
    coi = xr.DataArray(coi, dims=[dim], coords={dim: var[dim]})
    scale = xr.DataArray(scale, dims=["period"], coords={"period": period})

    w = xr.Dataset()
    w["coi"] = coi
    w["scale"] = scale
    w["wave"] = wave
    w["power"] = np.abs(w.wave) ** 2

    cmat, pmat = xr.broadcast(w.coi, w.period)
    w = w.where(pmat < cmat)
    w = w.dropna(dim="period", how="all")

    w.attrs["long_name"] = "Wavelet power"

    return w


def plot_scalogram(da, dt=1, ax=None, **kwargs):

    w = wavelet(da, dt)

    if ax is None:
        f, ax = plt.subplots()

    robust = kwargs.pop("robust", True)
    cmap = kwargs.pop("cmap", svc.cm.blue_orange_div)
    levels = kwargs.pop("levels", 20)

    (
        np.log10(w.power).plot.contourf(
            yscale="log",
            yincrease=False,
            ax=ax,
            cmap=cmap,
            robust=robust,
            center=False,
            levels=levels,
            **kwargs,
        )
    )

    (
        np.log10(w.power).plot.contour(
            yscale="log",
            yincrease=False,
            ax=ax,
            colors="k",
            robust=robust,
            center=False,
            linestyles="-",
            linewidths=1,
            levels=np.nanpercentile(np.log10(w.power), [90, 95]),
            **kwargs,
        )
    )


def plot_detailed_scalogram(da, dt=1, **kwargs):

    with plt.style.context("ggplot"):
        f, ax = plt.subplots(2, 1, sharex=True, constrained_layout=True)

        plot_scalogram(da, dt=dt, ax=ax[0])

        da.plot(ax=ax[1])

        ax[0].set_xlabel("")

    return ax


def matlab_wavelet(da, dt=1, beta=2.0, gamma=3.0, eng=None, kind="matlab"):

    import matlab.engine

    if eng is None:
        print("Starting MATLAB...")
        eng = matlab.engine.start_matlab()
        eng.addpath(eng.genpath("~/tools/"))

    if isinstance(da, xr.DataArray):
        if da.ndim > 1:
            raise ValueError("Only 1-D data supported!")

        dim = da.dims[0]
        nparray = da.values

    elif isinstance(da, np.ndarray):
        nparray = da

    marray = eng.transpose(matlab.double(list(nparray)))

    if kind == "jlab":
        fs = eng.morsespace(float(gamma), float(beta), float(len(da)))

        wa = np.asarray(eng.wavetrans(marray, [gamma, beta, eng.squeeze(fs)]))

        period = dt * 2 * np.pi / np.asarray(fs).squeeze()

        coi = None

    elif kind == "matlab":
        wa, f, coi = eng.cwt(
            marray, 1.0, "WaveletParameters", matlab.double([gamma, beta]), nargout=3
        )
        period = dt * 2 * np.pi / np.asarray(f).squeeze()

        coi = np.asarray(coi).squeeze()
        wa = np.asarray(wa).squeeze().T

    if isinstance(da, xr.DataArray):
        power = xr.DataArray(
            np.abs(wa) ** 2,
            dims=[dim, "period"],
            coords={dim: da[dim], "period": period},
        )

        w = xr.Dataset()
        w["period"] = xr.DataArray(period, dims=["period"], coords={"period": period})
        w["wave"] = xr.DataArray(
            wa, dims=[dim, "period"], coords={dim: da[dim], "period": period}
        )
        w["power"] = power

        if coi is not None:
            w["coi"] = xr.DataArray(coi, dims=[dim], coords={dim: da[dim]})

            cmat, pmat = xr.broadcast(w.coi, w.period)
            w = w.where(pmat < cmat)
            w = w.dropna(dim="period", how="all")

        w.attrs["long_name"] = "Wavelet power"

    else:
        w = np.abs(wa)

    return w


def blackman(y, half_width):
    """
    Simple Blackman filter.

    The end effects are handled by calculating the weighted
    average of however many points are available, rather than
    by zero-padding.
    """
    nf = half_width * 2 + 1
    x = np.linspace(-1, 1, nf, endpoint=True)
    x = x[1:-1]  # chop off the useless endpoints with zero weight
    w = 0.42 + 0.5 * np.cos(x * np.pi) + 0.08 * np.cos(x * 2 * np.pi)
    ytop = np.convolve(y, w, mode="same")
    ybot = np.convolve(np.ones_like(y), w, mode="same")

    return ytop / ybot


def complex_demodulate(
    ts,
    central_period,
    t=None,
    dim=None,
    dt=1,
    bw=0.1,
    cycles_per="D",
    debug=False,
    filt="butter",
):

    if isinstance(ts, xr.DataArray):
        if dim is None:
            dim = ts.dims[0]

        t = ts[dim]

    if _is_datetime_like(t):
        dt, t = _process_time(t, cycles_per=cycles_per)

    if dim is None:
        dim = "dim_0"

    iscomplex = ts.dtype.kind == "c"  # complex input
    harmonic = np.exp(-1j * 2 * np.pi / np.abs(central_period) * t)
    harmonic_ccw = harmonic.conj()
    product = harmonic * ts

    lfreq = np.abs(bw * dt / central_period)
    # print(str(lfreq) + 'cp' + cycles_per.lower())

    if filt == "blackman":
        amp = blackman(product, int(round(bw * abs(central_period) / dt)))
    elif filt == "butter":
        amp = LowPassButter(product.real, lfreq, order=1) + 1j * LowPassButter(
            product.imag, lfreq, order=1
        )

    if not iscomplex:
        recon = (amp * harmonic_ccw * 2).real
    else:
        product_ccw = harmonic_ccw * ts
        if filt == "blackman":
            ampcw = blackman(product_ccw, int(round(bw * abs(central_period) / dt)))
        elif filt == "butter":
            ampcw = LowPassButter(
                product_ccw.real, lfreq, order=1
            ) + 1j * LowPassButter(product_ccw.imag, lfreq, order=1)

        recon = amp * harmonic_ccw + ampcw * harmonic

    dm = xr.Dataset()
    dm["ccw"] = xr.DataArray(amp, dims=[dim])
    if iscomplex:
        dm["cw"] = xr.DataArray(ampcw, dims=[dim])
    dm["recon"] = xr.DataArray(recon, dims=[dim])
    dm["amp"] = xr.DataArray(np.abs(recon), dims=[dim])
    dm["pha"] = xr.DataArray(np.angle(recon, deg=True), dims=[dim])
    if isinstance(ts, xr.DataArray):
        dm["signal"] = ts
    else:
        dm["signal"] = xr.DataArray(ts, dims=[dim])

    if debug:
        f, ax = plt.subplots(2, 2, sharey=True, constrained_layout=True)
        kwargs = dict(multitaper=True)
        if iscomplex:
            PlotSpectrum(dm["signal"], dt=dt, ax=ax[0, :], **kwargs)
            PlotSpectrum(dm["signal"], dt=dt, ax=ax[1, :], color="lightgray", **kwargs)
            PlotSpectrum(
                dm["recon"].dropna(dim=dim), dt=dt, twoside=True, ax=ax[0, :], **kwargs
            )
        else:
            PlotSpectrum(dm["signal"], dt=dt, ax=ax[0, 0], **kwargs)
            PlotSpectrum(dm["signal"], dt=dt, ax=ax[0, 1], **kwargs)
            PlotSpectrum(
                dm["recon"].dropna(dim=dim), dt=dt, twoside=True, ax=ax[0, 0], **kwargs
            )
            PlotSpectrum(
                dm["recon"].dropna(dim=dim), dt=dt, twoside=True, ax=ax[0, 1], **kwargs
            )

        PlotSpectrum(product, dt=dt, ax=ax[1, :], **kwargs)
        if iscomplex:
            PlotSpectrum(product_ccw, dt=dt, ax=ax[1, :], **kwargs)

        linex([1 / central_period, lfreq], ax=ax.ravel())

        ax[0, 0].legend(["signal", "demodulated", "central_period", "low-pass"])
        ax[1, 0].legend(
            ["signal", "product_cw", "product_ccw", "central_period", "low-pass"]
        )

        # diff = dm.signal-dm.recon
        # f, ax = plt.subplots()
        # ax.plot(dm.signal.real)
        # ax.plot(dm.recon.real)
        # ax.plot(diff.real)

    return dm


def find_peaks(data, dim, debug=True):
    """Only returns peak indexes along dimension "dim"."""
    import scipy as sp

    def wrap_find_peaks(invar, kwargs={}):
        result = sp.signal.find_peaks(invar, **kwargs)[0]

        # pad with NaNs so that shape of returned object is invariant
        new_result = np.full_like(invar, fill_value=np.nan)
        new_result[: len(result)] = result
        return new_result

    indexes = xr.apply_ufunc(
        wrap_find_peaks,
        data,
        vectorize=True,  # loops with numpy over core dim?
        dask="parallelized",  # loop with dask
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],  # expect latitude dim in output
        output_dtypes=[np.float32],  # required for dask
    )

    squashed = indexes.dropna(dim, how="all").drop(dim).drop("variable")

    if debug is True:
        plt.figure()
        data.isel(period=1).plot.line(x=dim)
        idx = squashed.isel(period=1).dropna("latitude").astype(np.int32)
        linex(data[dim][idx])

    return squashed
