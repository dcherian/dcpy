from scipy import signal
import xarray as xr
import numpy as np
import functools
import mixsea
import gsw


def _check_good_segment(N2, N, f, debug=False):
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
        if debug:
            print("too much N2 variance")
        return False, 1

    if N2.mean() < 1e-9:
        if debug:
            print("too unstratified")
        return False, 2

    if N < 2 * f:
        # Kunze et al (2017): With the expectation that N > 2f
        # is a minimal frequency bandwidth to allow internal wave–wave interactions,
        # segments with ⟨N⟩ less than 2f were also excluded
        if debug:
            print("too little bandwidth; possibly no wave-wave interactions")
        return False, 3

    if N ** 2 < f ** 2:
        if debug:
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


def estimate_turb_segment(P, N2, lat, max_wavelength=256, debug=False, crit="mixsea"):
    # GM 75; Kunze et al (2017)
    E0 = 6.3e-5  # nondim spectral energy level
    jstar = 3  # peak mode number
    b = 1300  # m; stratification length scale
    N0 = 5.24e-3  # rad/s
    K_0 = 5e-6  # m²/s for a mixing efficiency of Γ=0.2
    ε_0 = 6.73e-10
    f30 = 7.3e-5  # rad/s or dcpy.oceans.coriolis(30)
    π = np.pi
    f = np.abs(2 * (2 * π / 86400) * np.sin(lat * π / 180))

    # shear strain ratio; Whalen et al (2015) use 3; Kunze et al (2017) use 7
    if crit in ["mixsea", "whalen"]:
        Rω = 3
    elif crit == "kunze":
        Rω = 7

   # TODO: fix for argo
    dp = np.diff(P).min()

    mask = np.isfinite(N2)
    N2fit = np.polyval(np.polyfit(P[mask], N2[mask], deg=2), P)
    N = np.sqrt(N2fit.mean()).item()

    h_Rω = shearstrain_Rω(Rω)
    L_Nf = latitude_Nf(N, f)
    kzstar = (π * jstar / b) * (N / N0)

    isgood, flag = _check_good_segment(N2, N, f, debug=debug)

    # TODO: check
    N2[~mask] = N2fit[~mask]

    # Whalen et al (2015) use mean(N2fit) in the denominator
    # Kunze et al (2017) use N2fit;
    # mixsea uses N2fit.mean() for Polynomial fits
    #          and N2fit for adiabatic levelling
    ξ = (N2 - N2fit) / N2fit.mean()

    if np.sum(~np.isfinite(ξ)) > 0:
        return np.nan

    # kz, psd = signal.periodogram(
    #    ξ.data, fs=2 * π / dp, window="hann", detrend="linear", scaling="density"
    # )
    _, _, psd, kz = mixsea.helpers.psd(ξ.data, 2, ffttype="t", detrend=True)
    # correct for first difference
    psd /= np.sinc(kz * dp / 2 / π) ** 2

    ξgm = np.pi * E0 * b / 2 * jstar * kz ** 2 / (kz + kzstar) ** 2
    ξgmvar = np.nan
    i0 = np.argmin(np.abs(1 / kz[1:] - max_wavelength / 2 / π))

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
            idx = np.argmin(np.abs(2 * π / kz[1:] - min_wavelength))
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

    scale = (ξvar / ξgmvar) ** 2 * h_Rω * L_Nf
    K = K_0 * scale
    ε = ε_0 * (N / N0) ** 2 * scale

    if debug:
        print(
            f"ξgmvar: {ξgmvar:.3f}, ξvar: {ξvar:.3f}, N2: {N2bar:.2e} K: {K:.2e}, ε: {ε:.2e}"
        )

    return K, ε, ξvar, ξgmvar, N ** 2, flag


def concat_lol(segments, name):
    lst = [
        np.atleast_1d(bin[name])
        if np.isscalar(bin[name]) or len(bin[name]) > 0
        else [np.nan]
        for bin in segments
    ]
    return np.concatenate(lst)


def mixsea_to_xarray(result):
    ds = xr.Dataset()
    ds["eps"] = ("depth_bin", result["eps_st"])
    ds["krho"] = ("depth_bin", result["krho_st"])
    ds.attrs["description"] = "strain parameterization"
    ds["psd"] = (("depth_bin", "m"), result["P_strain"])
    ds["psdgm"] = (("depth_bin", "m"), result["P_strain_gm"])
    ds["depth_bin"] = result["depth_bin"]
    ds["m"] = result["m"]
    for var in ["strain", "N2", "N2smooth"]:
        ds[var] = ("depth", concat_lol(result["strain"], var))
    ds["depth"] = ("depth", concat_lol(result["strain"], "segz"))
    ds["N2mean"] = ("depth_bin", concat_lol(result["strain"], "N2mean"))
    ds["ξvar"] = ("depth_bin", result["Int_st"])
    ds["ξvargm"] = ("depth_bin", result["Int_stgm"])

    if not ds.indexes["depth"].is_unique:
        ds = ds.isel(depth=~ds.indexes["depth"].duplicated())
    return ds.isel(depth=ds.depth.notnull())


def do_mixsea_shearstrain(profile, depth_bins):
    _, _, resultpf = mixsea.shearstrain.shearstrain(
        -1 * gsw.z_from_p(profile.pressure, profile.latitude),
        profile.ctd_temperature,
        profile.ctd_salinity,
        profile.longitude,
        profile.latitude,
        window_size=245,
        depth_bin=depth_bins,
        return_diagnostics=True,
        smooth="PF",
    )

    _, _, resultal = mixsea.shearstrain.shearstrain(
        -1 * gsw.z_from_p(profile.pressure, profile.latitude),
        profile.ctd_temperature,
        profile.ctd_salinity,
        profile.longitude,
        profile.latitude,
        window_size=245,
        depth_bin=depth_bins,
        return_diagnostics=True,
        smooth="AL",
    )

    ds = xr.concat([mixsea_to_xarray(resultpf), mixsea_to_xarray(resultal)], dim="kind")
    ds["kind"] = ["PF", "AL"]
    return ds
