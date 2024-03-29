import functools
import warnings

import cf_xarray as cfxr
import gsw
import gsw_xarray as gswxr
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy import signal

from . import oceans, plots

salt_criteria = {
    "sea_water_salinity": {
        "standard_name": "sea_water_salinity|sea_water_practical_salinity"
    }
}


def trim_mld_mode_water(profile, mode=True):
    """
    Follows Whalen's approach of using a threshold criterion
    first to identify MLD, trimming that; and then applying again
    to find mode water. I apply both T, σ criteria and pick the max depth.
    """

    def find_thresh_delta(delta, thresh):
        """Finds threshold in delta."""
        depth = delta.cf["Z"].cf.isel(Z=delta > thresh).data
        if len(depth) > 0:
            return depth[0]
        else:
            return None

    T = profile.cf.standard_names["sea_water_temperature"][0]
    σ = profile.cf.standard_names["sea_water_sigma_t"][0]

    assert (profile.cf["Z"] >= 0).all().data
    near_surf = profile.cf[[T, σ]].cf.sel(Z=10, method="nearest")
    delta = np.abs(profile - near_surf)

    Tmld = find_thresh_delta(delta[T], 0.2)
    σmld = find_thresh_delta(delta[σ], 0.03)

    if Tmld is not None and σmld is not None:
        trimmed = profile.cf.sel(Z=slice(max(Tmld, σmld), None))
    else:
        # profile is too short
        return profile.cf.sel(Z=slice(0, 0)).assign_coords(
            Tmode=np.nan, σmode=np.nan, Tmld=np.nan, σmld=np.nan
        )

    if mode:
        near_surf = trimmed.cf[[T, σ]].cf.isel(Z=0)
        delta = np.abs(trimmed - near_surf)
        Tmode = find_thresh_delta(delta[T], 0.2)
        σmode = find_thresh_delta(delta[σ], 0.03)
    else:
        Tmode = Tmld
        σmode = σmld

    return profile.cf.sel(Z=slice(max(Tmode, σmode), None)).assign_coords(
        Tmode=Tmode, σmode=σmode, Tmld=Tmld, σmld=σmld
    )


def results_to_xarray(results, profile, criteria):
    data_vars = {
        var: (("pressure", "criteria"), results[var])
        for var in [
            "ε",
            "Kρ",
            "ξvar",
            "ξvargm",
        ]
    }
    data_vars.update(
        {
            var: (("pressure",), results[var])
            for var in ["Tzlin", "Tzmean", "mean_dTdz_seg", "N2mean"]
        }
    )
    coords = {
        var: ("pressure", results[var]) for var in ["flag", "pressure", "npts", "γmean"]
    }
    coords.update(
        {
            "γ_bounds": (("pressure", "nbnds"), results["γbnds"]),
            "p_bounds": (("pressure", "nbnds"), results["pbnds"]),
            "criteria": (
                "criteria",
                criteria,
                {"description": "criteria used to find integrating range"},
            ),
        }
    )
    coords.update({k: v for k, v in profile.coords.items() if k not in coords})

    turb = xr.Dataset(data_vars, coords)

    turb.ε.attrs = {"long_name": "$ε$", "units": "W/kg"}
    turb.Kρ.attrs = {"long_name": "$K_ρ$", "units": "m²/s"}
    turb.ξvar.attrs = {"long_name": "$ξ$", "description": "strain variance"}
    turb.ξvargm.attrs = {"long_name": "$ξ_{gm}$", "description": "GM strain variance"}
    turb.N2mean.attrs = {
        "long_name": "$N²$",
        "description": "mean of quadratic fit of N² with pressure",
    }
    turb.Tzlin.attrs = {
        "long_name": "$T_z^{lin}$",
        "description": "linear fit of T vs pressure",
    }
    turb.Tzmean.attrs = {
        "long_name": "$T_z^{quad}$",
        "description": (
            "mean of quadratic fit of Tz with pressure; " "like N² fitting for strain"
        ),
    }
    turb.mean_dTdz_seg.attrs = {
        "description": "mean of dTdz values in segment",
        "long_name": "$⟨T_z⟩$",
    }
    turb.pressure.attrs = {
        "axis": "Z",
        "standard_name": "sea_water_pressure",
        "positive": "down",
        "bounds": "p_bounds",
    }
    turb.npts.attrs = {"description": "number of points in segment"}
    turb.γmean.attrs = {"bounds": "γ_bounds"}
    turb.γmean.attrs.update(profile.cf["neutral_density"].attrs)

    for var in ["Tmld", "σmld", "Tmode", "σmode"]:
        turb[var] = profile[var].data
        turb[var].attrs["units"] = "m"
        if "mld" in var:
            turb[var].attrs["description"] = f"Δ{var[0]} criterion applied first time"
        if "mode" in var:
            turb[var].attrs["description"] = f"Δ{var[0]} criterion applied second time"

    turb.flag.attrs = {
        "flag_values": [-2, -1, 1, 2, 3, 4, 5],
        "flag_meanings": (
            "too_coarse too_short N²_variance_too_high too_"
            "too_little_bandwidth no_internal_waves good_data"
        ),
    }

    turb["χ"] = 2 * turb.Kρ * turb.Tzmean**2
    turb.χ.attrs = {"long_name": "$χ$", "units": "°C²/s"}
    turb["KtTz"] = turb.Kρ * turb.Tzmean
    turb.KtTz.attrs = {"long_name": "$K_ρθ_z$", "units": "°Cm/s"}

    return turb


def choose_bins(pres, dz_segment):
    lefts = np.sort(
        np.hstack(
            [
                np.arange(1000 - dz_segment // 2, pres[0], -dz_segment // 2),
                np.arange(1000, pres[-1] + 1, dz_segment // 2),
            ]
        )
    )
    rights = lefts + dz_segment

    # lefts = np.arange(pres[0], pres[-1] + 1, dz_segment // 2)
    # rights = lefts + dz_segment

    return lefts, rights


def _check_good_segment(N2, N, f, debug=False):
    """
    Check whether segment is good.

    Return
    ------
    good: bool
    flag:
       -1. No data
        1. max(N²) - min(N²) > 1e-9 : Too much N² variance
        2. mean(N²) < 1e-9 : Too unstratified
        3. N < 2f: Too little bandwidth
    """

    if sum(np.isfinite(N2)) == 0:
        return False, -1

    delN2 = (N2.max() - N2.min()).item()
    if delN2 > 5e-4:
        if debug:
            print(f"too much N2 variance: {delN2}")
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

    if N**2 < f**2:
        if debug:
            print("no internal waves")
        return False, 4

    return True, 5


@functools.lru_cache
def latitude_Nf(N, f):
    N0 = 5.24e-3  # rad/s
    f30 = 7.3e-5  # rad/s or oceans.coriolis(30)
    return f / f30 * np.arccosh(N / f) / np.arccosh(N0 / f30)


@functools.lru_cache
def shearstrain_Rω(Rω):
    return 1 / 6 / np.sqrt(2) * Rω * (Rω + 1) / np.sqrt(Rω - 1)


def estimate_turb_segment(P, N2, lat, max_wavelength=256, debug=False, criteria=None):
    # GM 75; Kunze et al (2017)
    E0 = 6.3e-5  # nondim spectral energy level
    jstar = 3  # peak mode number
    b = 1300  # m; stratification length scale
    N0 = 5.24e-3  # rad/s
    K_0 = 5e-6  # m²/s for a mixing efficiency of Γ=0.2
    ε_0 = 6.73e-10
    π = np.pi
    f = np.abs(2 * (2 * π / 86400) * np.sin(lat * π / 180))

    # TODO: ensure approximately uniform spacing
    dp = np.diff(P).min()
    mask = np.isfinite(N2)
    if np.sum(mask) > 0:
        N2fit = np.polyval(np.polyfit(P[mask], N2[mask], deg=2), P)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", np.RankWarning)
            N = np.sqrt(N2fit.mean()).item()
    else:
        N = np.nan

    isgood, flag = _check_good_segment(N2, N, f, debug=debug)
    if not isgood:
        return np.nan, np.nan, np.nan, np.nan, N**2, flag

    # if debug:
    #     plt.figure()
    #     plt.plot(N2)
    #     plt.plot(N2fit)

    L_Nf = latitude_Nf(N, f)
    kzstar = (π * jstar / b) * (N / N0)

    # TODO: check
    N2[~mask] = N2fit[~mask]

    # Whalen et al (2015) use mean(N2fit) in the denominator
    # Kunze et al (2017) use N2fit;
    # mixsea uses N2fit.mean() for Polynomial fits
    #          and N2fit for adiabatic levelling
    ξ = (N2 - N2fit) / N2fit.mean()

    if np.sum(~np.isfinite(ξ)) > 0:
        return np.nan

    kz, psd = signal.periodogram(
        ξ.data, fs=2 * π / dp, window="hann", detrend="linear", scaling="density"
    )
    # _, _, psd, kz = mixsea.helpers.psd(ξ.data, 2, ffttype="t", detrend=True)
    # if debug:
    #    plt.loglog(kz, psd)

    # correct for first difference
    psd /= np.sinc(kz * dp / 2 / π) ** 2

    ξgm = np.pi * E0 * b / 2 * jstar * kz**2 / (kz + kzstar) ** 2
    ξgmvar = np.nan
    i0 = np.argmin(np.abs(1 / kz[1:] - max_wavelength / 2 / π))

    if criteria is None:
        criteria = ["kunze", "mixsea", "whalen", "whalen_7"]

    elif isinstance(criteria, str):
        criteria = [criteria]

    h_Rω = np.empty((len(criteria),))
    ξvar = np.empty((len(criteria),))
    ξgmvar = np.empty((len(criteria),))
    for cindex, crit in enumerate(criteria):
        # shear strain ratio; Whalen et al (2015) use 3; Kunze et al (2017) use 7
        if crit in ["mixsea", "whalen"]:
            Rω = 3
        elif crit in ["kunze", "whalen_7"]:
            Rω = 7
        h_Rω[cindex] = shearstrain_Rω(Rω)

        if crit == "kunze":
            # integrating the strain spectra S[ξ](k_z) from the
            # lowest resolved vertical wavenumber (λ_z = 256 m) to the
            # wavenumber where variance exceeds a threshold value
            # of 0.05, which, for a GM-level spectrum, corresponds to
            # λ_z ≈ 50 m, in part to avoid contamination by ship heave
            # near 10-m wavelengths
            for idx in range(i0, 127):
                ξvar[cindex] = np.trapz(psd[i0:idx], x=kz[i0:idx])
                if ξvar[cindex] > 0.05:
                    idxint = np.arange(i0, idx)
                    break
                # continue looping if spectrum is  "oversaturated"
            else:
                idxint = []

        elif "whalen" in crit:
            for min_wavelength in np.arange(10, 41, dp):
                idx = np.argmin(np.abs(2 * π / kz[1:] - min_wavelength))
                ξvar[cindex] = np.trapz(psd[i0:idx], x=kz[i0:idx])
                if ξvar[cindex] <= 0.2:
                    idxint = np.arange(i0, idx)
                    break
                # continue looping if spectrum is  "oversaturated"
            else:
                # always oversaturated
                idxint = []

        elif crit == "mixsea":
            import mixsea

            idxint, _ = mixsea.shearstrain.find_cutoff_wavenumber(psd, kz, 0.22)
            ξvar[cindex] = np.trapz(psd[idxint], x=kz[idxint])

        if idxint == []:
            ξgmvar[cindex] = np.nan
        else:
            ξgmvar[cindex] = np.trapz(ξgm[idxint], x=kz[idxint])

    ξvar[ξvar < 1e-6] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = (ξvar / ξgmvar) ** 2 * h_Rω * L_Nf
        K = K_0 * scale
        ε = ε_0 * (N / N0) ** 2 * scale

    if debug:
        with np.printoptions(precision=2):
            print(
                f"{P[0]:4.0f} — {P[-1]:4.0f}dbar: ξgmvar: {ξgmvar}, "
                f"ξvar: {ξvar}, N2: {N2fit.mean()} K: {K}, ε: {ε}"
            )

    return K, ε, ξvar, ξgmvar, N**2, flag


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


def do_mixsea_shearstrain(profile, dz_segment):
    import mixsea

    P = profile.cf["sea_water_pressure"]
    depth_bins = choose_bins(P.data, dz_segment)

    with cfxr.set_options(custom_criteria=salt_criteria):
        kwargs = dict(
            depth=-1 * gsw.z_from_p(P, profile.cf["latitude"]),
            t=profile.cf["sea_water_temperature"],
            SP=profile.cf["sea_water_salinity"],
            lon=profile.cf["longitude"],
            lat=profile.cf["latitude"],
            window_size=dz_segment,
            depth_bin=depth_bins[0],
            return_diagnostics=True,
        )
    _, _, resultpf = mixsea.shearstrain.nan_shearstrain(**kwargs, smooth="PF")
    _, _, resultal = mixsea.shearstrain.nan_shearstrain(**kwargs, smooth="AL")

    ds = xr.concat([mixsea_to_xarray(resultpf), mixsea_to_xarray(resultal)], dim="kind")
    ds["kind"] = ["PF", "AL"]
    return ds


def process_profile(profile, dz_segment=200, criteria=None, debug=False):
    with cfxr.set_options(custom_criteria=salt_criteria):
        if "sea_water_conservative_temperature" not in profile.cf:
            profile["SA"] = gswxr.SA_from_SP(
                profile.cf["sea_water_salinity"],
                profile.cf["sea_water_pressure"],
                profile.cf["longitude"],
                profile.cf["latitude"],
            )
            profile["CT"] = gswxr.CT_from_t(
                SA=profile.SA,
                t=profile.cf["sea_water_temperature"],
                p=profile.cf["sea_water_pressure"],
            )

        if "sea_water_sigma_t" not in profile.cf:
            profile["σ_θ"] = gswxr.sigma0(
                profile.cf["sea_water_absolute_salinity"],
                profile.cf["sea_water_conservative_temperature"],
            )
    if "neutral_density" not in profile.cf:
        profile["γ"] = oceans.neutral_density(profile)

    if debug:
        profile_original = profile

    profile = trim_mld_mode_water(profile)

    # with cfxr.set_options(custom_criteria=salt_criteria):
    #     profile.cf["sea_water_salinity"]
    # profile.cf["sea_water_temperature"]
    P = profile.cf["sea_water_pressure"]

    Zdim = profile.cf.axes["Z"][0]
    Z = profile[Zdim]
    gamma_name = profile.cf.standard_names["neutral_density"][0]
    CT_name = profile.cf.standard_names["sea_water_conservative_temperature"][0]
    SA_name = profile.cf.standard_names["sea_water_absolute_salinity"][0]
    latitude = profile.cf["latitude"].data

    if profile.sizes[Zdim] < 13 and debug:
        raise ValueError("empty")

    lefts, rights = choose_bins(Z.data, dz_segment)

    results = {
        var: np.full((len(lefts),), fill_value=np.nan)
        for var in [
            "N2mean",
            "γmean",
            "flag",
            "npts",
            "pressure",
            "Tzlin",
            "Tzmean",
            "mean_dTdz_seg",
        ]
    }

    if criteria is None:
        criteria = ["mixsea", "kunze", "whalen", "whalen_7"]
    elif isinstance(criteria, str):
        criteria = [criteria]

    for var in ["Kρ", "ε", "ξvar", "ξvargm"]:
        results[var] = np.full((len(lefts), len(criteria)), fill_value=np.nan)

    for var in ["γbnds", "pbnds"]:
        results[var] = np.full((len(lefts), 2), fill_value=np.nan)

    # N² calculation is the expensive step; do it only once
    N2full, pmid = gswxr.Nsquared(profile[SA_name], profile[CT_name], P, lat=latitude)
    N2Zdim = f"{P.name}_mid"
    N2full = xr.DataArray(
        N2full, dims=(N2Zdim,), coords={N2Zdim: (N2Zdim, pmid, {"axis": "Z"})}
    )
    P.attrs = profile.cf["sea_water_pressure"].attrs

    with xr.set_options(keep_attrs=True):
        dTdz_ = -1 * profile[CT_name].cf.diff("Z") / P.cf.diff("Z")
        dTdzfull = dTdz_.assign_coords({Zdim: N2full.cf["Z"].data})
        dTdzfull[Zdim].attrs["axis"] = "Z"

    for idx, (l, r) in enumerate(zip(lefts, rights)):  # noqa
        seg = profile.cf.sel(Z=slice(l, r))

        if seg.sizes[Zdim] == 1:
            results["flag"][idx] = -1
            continue

        P = seg[Zdim]
        γ = seg[gamma_name]

        # Argo in NATRE region: At 1000m sampling dz changes drastically;
        # it helps to just drop that one point
        seg = seg.where(P.diff(Zdim) < 21, drop=True)

        results["npts"][idx] = seg.sizes[Zdim]

        # max dz of 20m; ensure min number of points
        if results["npts"][idx] < np.ceil(dz_segment / 20):
            results["flag"][idx] = -1
            continue

        # TODO: despike
        # TODO: unrealistic values
        N2 = N2full.cf.sel(Z=slice(P[0], P[-1]))

        # TODO: Is this interpolation sensible?
        dp = P.diff(Zdim)
        Pn2 = N2[N2Zdim]
        if dp.max() - dp.min() > 2:
            dp = dp.median()
            seg = seg.cf.interp(Z=np.arange(P[0], P[-1], dp.median()))
            N2 = N2.cf.interp(Z=np.arange(Pn2[0], Pn2[-1], dp.median()))
            Pn2 = N2[N2Zdim]

        results["pressure"][idx] = (P.data[0] + P.data[-1]) / 2
        results["pbnds"][idx, 0] = P.data[0]
        results["pbnds"][idx, 1] = P.data[-1]

        results["γmean"][idx] = γ.mean()
        results["γbnds"][idx, 0] = γ.data[0]
        results["γbnds"][idx, 1] = γ.data[-1]

        (
            results["Kρ"][idx],
            results["ε"][idx],
            results["ξvar"][idx],
            results["ξvargm"][idx],
            results["N2mean"][idx],
            results["flag"][idx],
        ) = estimate_turb_segment(
            Pn2.data,
            N2.data,
            latitude,
            max_wavelength=dz_segment,
            debug=debug,
            criteria=criteria,
        )

        results["Tzlin"][idx] = (
            seg[CT_name].polyfit(Zdim, deg=1).sel(degree=1).polyfit_coefficients.values
            * -1
        )
        dTdz = dTdzfull.cf.sel(Z=slice(l, r))

        results["mean_dTdz_seg"][idx] = dTdz.cf.mean("Z")

        dTdz_fit = xr.polyval(Pn2, dTdz.polyfit(Zdim, deg=2).polyfit_coefficients)
        results["Tzmean"][idx] = dTdz_fit.mean().data

        # if debug:
        #     import matplotlib.pyplot as plt

        #     plt.figure()
        #     dTdz.plot()
        #     dTdz_fit.plot()

    dataset = results_to_xarray(results, profile, criteria=criteria)

    if debug:
        plot_profile_turb(profile_original, dataset)

    return dataset


def plot_profile_turb(profile, result):
    if result.ε.isnull().data.all():
        print("no output!")
        return

    if "criteria" in result.sizes and result.sizes["criteria"] > 1:
        raise ValueError("Cannot plot for multiple criteria.")
    p_edges = cfxr.bounds_to_vertices(result.p_bounds, bounds_dim="nbnds")

    f, axx = plt.subplots(1, 4, sharey=True)

    ax = dict(zip(["T", "ξ", "strat", "turb"], axx.flat))
    xlabels = ["$T$", "$ξ$ var", "", ""]

    ax["S"] = ax["T"].twiny()
    ax["γ"] = ax["T"].twiny()
    plots.set_axes_color(ax["S"], "r")
    plots.set_axes_color(ax["γ"], "teal")

    profile.cf["sea_water_conservative_temperature"].cf.plot(
        ax=ax["T"], marker=".", markersize=4
    )
    profile.cf["sea_water_absolute_salinity"].cf.plot(
        ax=ax["S"], color="r", _labels=False
    )
    profile.cf["neutral_density"].cf.plot(ax=ax["γ"], color="teal", _labels=False)

    title = ax["T"].get_title()
    [a.set_title("") for a in axx.flat]

    result = result.cf.sel(Z=result.cf["Z"].notnull())
    result.ξvar.cf.plot(ax=ax["ξ"], _labels=False)
    result.ξvargm.cf.plot(ax=ax["ξ"], _labels=False)
    ax["ξ"].legend(["obs", "GM"])

    (9.81 * 1.7e-4 * result.Tzmean).cf.plot(ax=ax["strat"], _labels=False)
    result.N2mean.cf.plot(ax=ax["strat"], _labels=False)
    ax["strat"].legend(["$gαT_z$", "$N^2$"])

    result.ε.cf.plot(ax=ax["turb"], _labels=False)
    result.χ.cf.plot(ax=ax["turb"], _labels=False, xscale="log")
    ax["turb"].legend(["χ", "ε"])

    plots.liney([result.Tmld, result.Tmode], color="k", ax=axx.flat)
    plots.liney([result.σmld, result.σmode], color="b", ax=axx.flat)

    for lab, a in zip(xlabels, axx.flat):
        a.set_xlabel(lab)
        a.set_yticks(p_edges, minor=True)
        a.grid(True, axis="y", which="minor")

    f.suptitle(title)


def process_argo_profile(profile, dz_segment=200, criteria=None, debug=False):
    """
    Processes finestructure turbulence estimate for Argo profiles
    in half-overlapping segments of length dz_segment.

    Parameters
    ----------

    profile: xr.DataArray
        Argo profile.
    dz_segment: optional
        Length of segment in dbar.
    criteria: hashable or sequence of hashable, optional
        Passed to process_profile.
    Returns
    -------

    xr.Dataset if profile is not bad or too short.
    """

    for var in ["PRES", "TEMP", "PSAL"]:
        if profile[f"{var}_QC"] != 1:
            if debug:
                raise ValueError("bad_quality")
            return ["bad_quality"]

    profile = profile.isel(N_LEVELS=profile.PRES.notnull()).swap_dims(
        {"N_LEVELS": "PRES"}
    )

    dataset = process_profile(profile, dz_segment, criteria)

    for var in [
        "CONFIG_MISSION_NUMBER",
        "PLATFORM_NUMBER",
        "CYCLE_NUMBER",
        "DIRECTION",
    ]:
        dataset.coords[var] = profile[var].data

    return dataset
