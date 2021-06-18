from scipy import signal
import xarray as xr
import numpy as np
import functools
import mixsea


def _check_good_segment(N2, N, f):
    """
    Check whether segment is good.

    Return
    ------
    good: bool
    flag:
        1. max(N²) - min(N²) > 1e-9 : Too much N² variance
        2. mean(N²) < 1e-9 : Too unstratified
        3. N < 2f: Too little bandwidth
    """

    if (N2.max() - N2.min()).item() > 5e-4:
        print("too much N2 variance")
        return False, 1

    if N2.mean() < 1e-9:
        print("too unstratified")
        return False, 2

    if N < 2 * f:
        # Kunze et al (2017): With the expectation that N > 2f
        # is a minimal frequency bandwidth to allow internal wave–wave interactions,
        # segments with ⟨N⟩ less than 2f were also excluded
        print("too little bandwidth; possibly no wave-wave interactions")
        return False, 3

    if N ** 2 < f ** 2:
        print("no internal waves")
        return False, 4

    return True, 5


@functools.lru_cache
def latitude_Nf(N, f):
    N0 = 5.24e-3  # rad/s
    f30 = 7.3e-5  # rad/s or dcpy.oceans.coriolis(30)
    return f / f30 * np.arccosh(N / f) / np.arccosh(N0 / f30)


@functools.lru_cache
def shearstrain_Rω(Rω):
    return 1 / 6 / np.sqrt(2) * Rω * (Rω + 1) / np.sqrt(Rω - 1)


def estimate_turb_segment(seg, debug=False):
    # GM 75; Kunze et al (2017)
    E0 = 6.3e-5  # nondim spectral energy level
    jstar = 3  # peak mode number
    b = 1300  # m; stratification length scale
    N0 = 5.24e-3  # rad/s
    K_0 = 5e-6  # m²/s for a mixing efficiency of Γ=0.2
    eps_0 = 6.73e-10
    f30 = 7.3e-5  # rad/s or dcpy.oceans.coriolis(30)
    π = np.pi

    crit = "mixsea"
    # shear strain ratio; Whalen et al (2015) use 3; Kunze et al (2017) use 7
    if crit in ["mixsea", "whalen"]:
        Rω = 3
    elif crit == "kunze":
        Rω = 7

    # TODO: fix for argo
    P = seg.cf["sea_water_pressure"]
    dp = np.diff(P.data).min()

    N2 = seg.N2
    mask = N2.notnull()
    N2fit = xr.DataArray(
        dims=N2.dims,
        data=np.polyval(np.polyfit(P[mask], N2[mask], deg=2), P),
    )
    if debug:
        N2.cf.plot(y="sea_water_pressure")
        N2fit.cf.plot(y="sea_water_pressure")

    lat = seg.cf["latitude"].item()
    f = np.abs(2 * (2 * π / 86400) * np.sin(lat * π / 180))
    N = np.sqrt(N2fit.mean()).item()

    isgood, flag = _check_good_segment(N2, N, f)

    # TODO: check
    N2.data[~mask] = N2fit.data[~mask]

    # Whalen et al (2015) use mean(N2fit) in the denominator
    # Kunze et al (2017) use N2fit;
    # mixsea uses N2fit.mean() for Polynomial fits
    #          and N2fit for adiabatic levelling
    ξ = (N2 - N2fit) / N2fit.mean()

    assert ξ.isnull().sum().item() == 0

    kz, psd = signal.periodogram(
        ξ.data, fs=2 * π / dp, window="hann", detrend="linear", scaling="density"
    )
    _, _, psd, kz = mixsea.helpers.psd(ξ.data, 2, ffttype="t", detrend=True)
    # correct for first difference
    psd /= np.sinc(kz * dp / 2 / π) ** 2

    h_Rω = shearstrain_Rω(Rω)
    L_Nf = latitude_Nf(N, f)
    kzstar = (π * jstar / b) * (N / N0)
    ξgm = np.pi * E0 * b / 2 * jstar * kz ** 2 / (kz + kzstar) ** 2

    ξgmvar = np.nan
    i0 = np.argmin(np.abs(1 / kz[1:] - 256 / 2 / π)) + 1

    if crit == "kunze":
        # TODO: integrating the strain spectra S[ξ](k_z) from the
        # lowest resolved vertical wavenumber (λ_z = 256 m) to the
        # wavenumber where variance exceeds a threshold value
        # of 0.05, which, for a GM-level spectrum, corresponds to
        # λ_z ≈ 50 m, in part to avoid contamination by ship heave
        # near 10-m wavelengths
        for idx in range(i0, 127):
            ξvar = np.trapz(psd[i0:idx], x=kz[i0:idx])
            if ξvar > 0.05:
                idxint = np.arange(i0, idx + 1)
                break
            # continue looping if spectrum is  "oversaturated"

    elif crit == "whalen":
        for min_wavelength in np.arange(10, 41, dp):
            idx = np.argmin(np.abs(2 * π / kz[1:] - min_wavelength)) + 1
            ξvar = np.trapz(psd[i0:idx], x=kz[i0:idx])
            if ξvar <= 0.2:
                idxint = np.arange(i0, idx + 1)
                break
            # continue looping if spectrum is  "oversaturated"

    elif crit == "mixsea":
        idxint, _ = mixsea.shearstrain.find_cutoff_wavenumber(psd, kz, 0.22)
        ξvar = np.trapz(psd[idxint], x=kz[idxint])

    ξgmvar = np.trapz(ξgm[idxint], x=kz[idxint])

    if np.isnan(ξgmvar):
        ξvar = np.nan

    K = K_0 * ξvar ** 2 / ξgmvar ** 2 * h_Rω * L_Nf
    ε = eps_0 * N ** 2 / N0 ** 2 * ξvar ** 2 / ξgmvar ** 2 * h_Rω * L_Nf
    if debug:
        print(
            f"ξgmvar: {ξgmvar:.3f}, ξvar: {ξvar:.3f}, N2: {N2bar:.2e} K: {K:.2e}, ε: {ε:.2e}"
        )

    return K, ε, ξvar, ξgmvar
