import os

import cf_xarray as cfxr  # noqa
import matplotlib as mpl
import matplotlib.pyplot as plt
import netCDF4 as nc
import numba
import numpy as np
import pandas as pd
import seawater as sw
import xarray as xr

from . import eos, plots, util


def _flatten_data(data):
    if isinstance(data, xr.DataArray):
        d = data.values.ravel()
    elif isinstance(data, np.ndarray):
        d = data.ravel()
    else:
        return data

    return d[~np.isnan(d)]


def dataset_center_pacific(da, name=None):
    """Takes an input DataArray and rolls the longitude
    so that all 3 basins are covered. Longitude goes from 20 to 380."""

    if name is None:
        name = "longitude" if "longitude" in da.coords else "lon"

    # roll so that pacific is in the middle
    # and we have coverage of all 3 basins

    da = da.roll(**{name: -1 * da[name].searchsorted(20)}, roll_coords=True)

    coord = da[name].values
    coord[coord < 20] += 360

    da = da.assign_coords({name: coord})

    return da


def coriolis(lat):
    π = np.pi
    return 2 * (2 * π / 86400) * np.sin(lat * π / 180)


def ReadWoa(lon, lat, time="annual", depth=None, return_xr=False):
    """Given lon, lat and type, return WOA data.
    Input:
          lon : +ve East
          lat : +ve North
          time: 'annual' or 'seasonal'
          depth: float (m)
    Output:
          Returns a dictionary with T, S, depth
    """

    woa = dict()

    # read in World Ocean Atlas data
    if time == "annual":
        woaT = nc.Dataset("/home/deepak/datasets/woa13_decav_t00_01v2.nc", "r")
        woaS = nc.Dataset("/home/deepak/datasets/woa13_decav_s00_01v2.nc", "r")

    if time == "seasonal":
        woaT = nc.MFDataset(
            "/home/deepak/datasets/woa13-season/" + "woa13_decav_t*_01v2.nc",
            "r",
            aggdim="time",
        )
        woaS = nc.MFDataset(
            "/home/deepak/datasets/woa13-season/" + "woa13_decav_s*_01v2.nc",
            "r",
            aggdim="time",
        )

    latind = np.where(woaT["lat"][:] < lat)[0][-1]
    lonind = np.where(woaT["lon"][:] < lon)[0][-1]

    def ConcatAndAverage(variable, lonind, latind):
        shape = list(variable[:, :, latind, lonind].shape)
        shape.append(1)

        avg = np.concatenate(
            (
                np.reshape(variable[:, :, latind, lonind], shape),
                np.reshape(variable[:, :, latind, lonind + 1], shape),
                np.reshape(variable[:, :, latind + 1, lonind], shape),
                np.reshape(variable[:, :, latind + 1, lonind + 1], shape),
            ),
            axis=len(shape) - 1,
        )
        avg[avg > 50] = np.nan
        return np.nanmean(avg, axis=len(shape) - 1)

    woa["T"] = ConcatAndAverage(woaT["t_an"], lonind, latind)
    woa["S"] = ConcatAndAverage(woaS["s_an"], lonind, latind)
    woa["depth"] = woaT["depth"][:]

    if depth is not None:
        index = np.where(woaT["depth"][:] == np.abs(depth))
        woa["depth"] = abs(depth)
        woa["T"] = woa["T"][:, index]
        woa["S"] = woa["S"][:, index]

    if return_xr:
        import xarray as xr

        woadict = woa

        woa = xr.Dataset(
            {
                "T": (["depth"], np.squeeze(woadict["T"])),
                "S": (["depth"], np.squeeze(woadict["S"])),
            },
            coords={"depth": (["depth"], woadict["depth"])},
        )
    return woa


def GM(lat, N, N0, b=1000, oned=False):
    try:
        import GM81.gm as gm
    except ImportError:
        raise ImportError("Please install the GM81 package.")

    # Coriolis frequency
    f = sw.f(lat=12)

    # frequency
    omg = np.logspace(np.log10(1.01 * f), np.log10(N), 401)

    # horizontal wavenumber
    k = 2 * np.pi * np.logspace(-6, -2, 401)

    # mode number
    j = np.arange(1, 100)

    # reshape to allow multiplication into 2D array
    Omg = np.reshape(omg, (omg.size, 1))
    K = np.reshape(k, (k.size, 1))
    J = np.reshape(j, (1, j.size))

    # frequency spectra (KE and PE)
    K_omg_j = gm.K_omg_j(Omg, J, f, N, N0, b)
    P_omg_j = gm.P_omg_j(Omg, J, f, N, N0, b)

    # wavenumber spectra (KE and PE)
    K_k_j = gm.K_k_j(K, J, f, N, N0, b)
    P_k_j = gm.P_k_j(K, J, f, N, N0, b)

    # sum over modes
    K_omg = np.sum(K_omg_j, axis=1)
    P_omg = np.sum(P_omg_j, axis=1)
    K_k = np.sum(K_k_j, axis=1)
    P_k = np.sum(P_k_j, axis=1)

    # compute 1D spectra from 2D spectra
    K_k_1d = gm.calc_1d(k, K_k)
    P_k_1d = gm.calc_1d(k, P_k)

    return (omg, K_omg, P_omg, k, K_k_1d, P_k_1d)


def TSplot(
    S,
    T,
    Pref=0,
    size=None,
    color=None,
    ax=None,
    rho_levels=[],
    labels=True,
    label_spines=True,
    plot_distrib=True,
    Sbins=30,
    Tbins=30,
    plot_kwargs=None,
    hexbin=True,
    fontsize=9,
    kind=None,
    equalize=True,
):
    """
    T-S plot. The default is to scatter, but hex-binning is also an option.

    Parameters
    ----------
    S, T : float32
        Salinity, temperature.
    Pref : float32, optional
        Reference pressure level.
    size: int, numpy.ndarray, xr.DataArray
        Passed to scatter.
    color: string, numpy.ndarray, xr.DataArray
        Passed as 'c' to scatter and 'C' to hexbin.
    ax : optional, matplotlib.Axes
        Axes to plot to.
    rho_levels : optional
        Density contour levels.
    labels : bool, optional
        Label density contours?
    label_spines : bool, optional
        Fancy spine labelling inspired by Arnold Gordon's plots.
    fontsize: int, optional
        font size for labels
    plot_distrib : bool, optional
        Plot marginal distributions of T, S?
    Sbins, Tbins : int, optional
        Number of T, S bins for marginal distributions.
    hexbin : bool, optional
        hexbin instead of scatter plot?
    plot_kwargs: dict, optional
        extra kwargs passed directly to scatter or hexbin. Cannot contain
        's', 'c' or 'C'.

    Returns
    -------
    handles: dict,
        cs : Handle to density ContourSet.
        ts : Handle to T-S scatter
        Thist, Shist : handles to marginal distributions
    axes : list of Axes.
    """

    # colormap = cmo.cm.matter
    #
    if kind is None:
        if hexbin is True:
            kind = "hexbin"
        elif hexbin is False:
            kind = "scatter"

    if plot_kwargs is None:
        plot_kwargs = {}
    if any([kw in plot_kwargs for kw in ["c", "C", "s"]]):
        raise ValueError(
            "plot_kwargs cannot contain c, C, or s. "
            + "Please specify size or color as appropriate."
        )

    scatter_defaults = {"edgecolors": None, "alpha": 0.5}

    labels = False if rho_levels is None else labels

    axes = dict()
    handles = dict()

    if ax is None:
        f = plt.figure(constrained_layout=True)
        if plot_distrib:
            gs = mpl.gridspec.GridSpec(5, 5, figure=f)

            axes["ts"] = f.add_subplot(gs[1:, :-1])
            axes["s"] = f.add_subplot(gs[0, :-1], sharex=axes["ts"])
            axes["t"] = f.add_subplot(gs[1:, -1], sharey=axes["ts"])
            ax = axes["ts"]
        else:
            ax = plt.gca()
    elif isinstance(ax, dict):
        axes = ax
        ax = axes["ts"]
    axes["ts"] = ax

    nanmask = np.isnan(S.values) | np.isnan(T.values)
    if size is not None:
        nanmask = nanmask | np.isnan(size)
    if color is not None and not isinstance(color, str):
        nanmask = nanmask | np.isnan(color)
    if len(np.atleast_1d(Sbins)) > 1:
        nanmask = nanmask | (S.values < np.min(Sbins)) | (S.values > np.max(Sbins))
    if len(np.atleast_1d(Tbins)) > 1:
        nanmask = nanmask | (T.values < np.min(Tbins)) | (T.values > np.max(Tbins))

    salt = _flatten_data(S.where(~nanmask))
    temp = _flatten_data(T.where(~nanmask))
    if size is not None and hasattr(size, "values"):
        size = _flatten_data(size.where(~nanmask))
    if color is not None and hasattr(color, "values"):
        color = _flatten_data(color.where(~nanmask))

    # TODO: plot outliers separately with hexbin
    # _prctile = 2
    # outlierT = np.percentile(temp, [_prctile, 100 - _prctile])
    # outlierS = np.percentile(salt, [_prctile, 100 - _prctile])

    # outliermask = np.logical_or(
    #     np.logical_or(salt > outlierS[1], salt < outlierS[0]),
    #     np.logical_or(temp > outlierT[1], temp < outlierT[0]))

    if kind == "hexbin":

        hexbin_defaults = {"cmap": mpl.cm.Blues, "mincnt": 1}
        if not isinstance(color, str):
            hexbin_defaults["C"] = color
        hexbin_defaults.update(plot_kwargs)

        handles["ts"] = ax.hexbin(
            salt, temp, gridsize=(Sbins, Tbins), **hexbin_defaults
        )
        # ax.plot(salt[outliermask], temp[outliermask], '.', 'gray')
        #
    elif kind == "hist":
        from xhistogram.core import histogram

        if isinstance(Sbins, int):
            Sbins = np.linspace(S.data.min(), S.data.max(), Sbins)
        if isinstance(Tbins, int):
            Tbins = np.linspace(T.data.min(), T.data.max(), Tbins)
        hist = histogram(salt, temp, bins=(Sbins, Tbins))
        if equalize:
            from skimage.exposure import equalize_adapthist

            hist = equalize_adapthist(hist.data)
        hist = hist.astype(float)
        # print(np.percentile(hist.ravel(), 10))
        hist[hist < np.percentile(hist[hist > 0].ravel(), 10)] = np.nan
        handles["ts"] = ax.pcolormesh(Sbins, Tbins, hist.T, **plot_kwargs)

    elif kind == "scatter":
        scatter_defaults.update(plot_kwargs)
        handles["ts"] = ax.scatter(
            salt,
            temp,
            s=size if size is not None else 12,
            c=color,
            **scatter_defaults,
        )

    # defaults.pop('alpha')
    # ts = ax.scatter(flatten_data(S), flatten_data(T),
    #                 s=flatten_data(size), c=[[0, 0, 0, 0]],
    #                 **defaults)

    if rho_levels is not None:
        Slim = ax.get_xlim()
        Tlim = ax.get_ylim()

        Tvec = np.linspace(Tlim[0], Tlim[1], 40)
        Svec = np.linspace(Slim[0], Slim[1], 40)
        [Smat, Tmat] = np.meshgrid(Svec, Tvec)

        # background ρ contours are T, S at the reference level
        ρ = sw.pden(Smat, Tmat, Pref, Pref) - 1000

        rho_levels = np.asarray(rho_levels)
        if np.all(rho_levels > 1000):
            rho_levels = rho_levels - 1000
        if not (rho_levels.size > 0):
            rho_levels = 7

        handles["rho_contours"] = ax.contour(
            Smat,
            Tmat,
            ρ,
            colors="gray",
            levels=rho_levels,
            linestyles="solid",
            zorder=-1,
            linewidths=0.5,
        )

        ax.set_xlim([Smat.min(), Smat.max()])
        ax.set_ylim([Tmat.min(), Tmat.max()])

    if plot_distrib:
        hist_args = dict(color=color, density=True, histtype="step")
        handles["Thist"] = axes["t"].hist(
            temp, orientation="horizontal", bins=Tbins, **hist_args
        )
        axes["t"].set_xticklabels([])
        axes["t"].set_xticks([])
        axes["t"].spines["bottom"].set_visible(False)

        handles["Shist"] = axes["s"].hist(salt, bins=Sbins, **hist_args)
        axes["s"].set_yticks([])
        axes["s"].set_yticklabels([])
        axes["s"].spines["left"].set_visible(False)

    if labels:
        if label_spines:
            plots.contour_label_spines(
                handles["rho_contours"], fmt="%.1f", fontsize=fontsize
            )
        else:
            clabels = ax.clabel(
                handles["cs"], fmt="%.1f", inline=True, inline_spacing=10
            )
            [txt.set_backgroundcolor([0.95, 0.95, 0.95, 0.75]) for txt in clabels]

        ax.text(
            1.005,
            1.00,
            " $σ_{" + str(Pref) + "}$",
            transform=ax.transAxes,
            va="top",
            fontsize=fontsize + 2,
            color="gray",
        )

    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_visible(True)

    ax.set_xlabel("S")
    ax.set_ylabel("T")

    return handles, axes


def argo_mld_clim(kind="monthly", fname=None):
    import glob
    import os

    if fname is None:
        if kind == "monthly":
            loc = "~/datasets/argomld/Argo_mixedlayers_monthlyclim_*.nc"
        if kind == "annual":
            loc = "~/datasets/argomld/Argo_mixedlayers_all_*.nc"
        loc = os.path.expanduser(loc)
        fname = glob.glob(loc)

    if not fname:
        raise ValueError(f"No files found at {loc}!")

    if len(fname) > 1:
        raise ValueError("Multiple files found. Either delete one or pass ``fname``")

    ds = xr.open_dataset(fname[0])

    mld = xr.Dataset(
        coords={
            "lat": ds["lat"].rename({"iLAT": "lat"}),
            "lon": ds["lon"].rename({"iLON": "lon"}),
            "month": ds["month"].rename({"iMONTH": "month"}),
        }
    )

    for da in ds:
        name = ""
        if da[0:2] == "ml":
            mld[da] = (("lat", "lon", "month"), ds[da].data)

            if "mean" in da:
                name = "Mean"

            if "median" in da:
                name = "Median"

            if "std" in da:
                name = "Std. dev. of"

            if "max" in da:
                name = "Max"

            if "mld" in da:
                name += " mixed layer depth"
                mld[da].attrs["units"] = "m"

            if "mlpd" in da:
                name = "Mean potential density using MLD"
                mld[da].attrs["units"] = "kg/m^3"

            if "mlt_" in da:
                name = "Mean temperature using MLD"
                mld[da].attrs["units"] = "celsius"

            if "mls_" in da:
                name = "Mean salinity using MLD"

            if "_da_" in da:
                name += " from density algorithm"

            if "_dt_" in da:
                name += " from density threshold algorithm"

            mld[da].attrs["long_name"] = name
            mld[da].attrs["positive"] = "down"
    mld.month.attrs["axis"] = "T"
    mld.month.attrs["_CoordinateAxisType"] = "Time"
    return mld


def read_trmm(dirname="../datasets/trmm/3B42_Daily.*.nc4.nc4"):
    def preprocess(ds):
        ds["time"] = pd.to_datetime(ds.attrs["BeginDate"] + " " + ds.attrs["BeginTime"])
        return ds

    trmm = xr.open_mfdataset(dirname, preprocess=preprocess, concat_dim="time")
    trmm.attrs["units"] = "mm/day"

    return trmm.transpose()


def read_imerg():
    def preprocess(ds):
        ds["time"] = pd.to_datetime(ds.attrs["BeginDate"] + " " + ds.attrs["BeginTime"])
        ds = ds.expand_dims("time")

        return ds

    imerg = xr.open_mfdataset(
        "../datasets/imerg/3B-DAY-E*.nc4.nc4", preprocess=preprocess, concat_dim="time"
    )

    return imerg.transpose()


def read_aquarius(dirname="/home/deepak/datasets/aquarius/oisss/"):

    import cftime

    def preprocess(ds):
        t = np.datetime64(
            cftime.num2date(ds.time[0], "days since 2010-12-31", calendar="julian")
        ) + np.timedelta64(3, "D")

        dsnew = ds.squeeze().drop("time")
        dsnew["time"] = xr.DataArray([t], dims=["time"])
        dsnew["sss"] = dsnew.sss.expand_dims("time")

        return dsnew

    aq = xr.open_mfdataset(dirname + "*.nc", preprocess=preprocess, decode_times=False)

    return aq


def read_aquarius_l3(dirname="/home/deepak/datasets/aquarius/L3/combined/"):
    def preprocess(ds):

        dsnew = ds.copy()
        dsnew["latitude"] = xr.DataArray(
            np.linspace(90, -90, 180), dims=["phony_dim_0"]
        )
        dsnew["longitude"] = xr.DataArray(
            np.linspace(-180, 180, 360), dims=["phony_dim_1"]
        )
        dsnew = (
            dsnew.rename(
                {
                    "l3m_data": "sss",
                    "phony_dim_0": "latitude",
                    "phony_dim_1": "longitude",
                }
            )
            .set_coords(["latitude", "longitude"])
            .drop("palette")
        )

        dsnew["time"] = (
            pd.to_datetime(dsnew.attrs["time_coverage_start"])
            + np.timedelta64(3, "D")
            + np.timedelta64(12, "h")
        )
        dsnew = dsnew.expand_dims("time").set_coords("time")

        return dsnew

    aq = xr.open_mfdataset(dirname + "/*.nc", preprocess=preprocess)

    aq["latitude"].attrs["units"] = "degrees_east"
    aq["longitude"].attrs["units"] = "degrees_north"
    aq.sss.attrs["long_name"] = "Sea Surface Salinity"

    aq = aq.sortby("latitude")

    return aq


def read_argo_clim(dirname="/home/deepak/datasets/argoclim/", chunks=None):

    if chunks is None:
        chunks = {"LATITUDE": 20, "LONGITUDE": 60}

    argoT = xr.open_mfdataset(
        dirname + "RG_ArgoClim_Temperature_*.nc", decode_times=False, chunks=chunks
    )
    argoS = xr.open_mfdataset(
        dirname + "RG_ArgoClim_Salinity_*.nc", decode_times=False, chunks=chunks
    )

    argoS["S"] = argoS.ARGO_SALINITY_ANOMALY + argoS.ARGO_SALINITY_MEAN
    argoT["T"] = argoT.ARGO_TEMPERATURE_ANOMALY + argoT.ARGO_TEMPERATURE_MEAN

    # override for BATHYMETRY_MASK which is in both files
    argo = xr.merge([argoT, argoS], compat="override")

    argo = argo.rename(
        {
            "LATITUDE": "lat",
            "LONGITUDE": "lon",
            "PRESSURE": "pres",
            "TIME": "time",
            "ARGO_TEMPERATURE_MEAN": "Tmean",
            "ARGO_TEMPERATURE_ANOMALY": "Tanom",
            "ARGO_SALINITY_MEAN": "Smean",
            "ARGO_SALINITY_ANOMALY": "Sanom",
        }
    )

    _, ref_date = xr.coding.times._unpack_netcdf_time_units(argo.time.units)

    argo = argo.assign_coords(
        time=pd.Timestamp(ref_date) + pd.to_timedelta(30 * argo.time.values, unit="D")
    )

    argo.time.attrs["axis"] = "T"
    argo["T"].attrs["standard_name"] = "sea_water_temperature"
    argo["Tmean"].attrs["standard_name"] = "sea_water_temperature"
    argo["S"].attrs["standard_name"] = "sea_water_salinity"
    argo["Smean"].attrs["standard_name"] = "sea_water_salinity"
    argo["pres"].attrs["standard_name"] = "sea_water_pressure"
    argo.pres.attrs["positive"] = "down"
    argo["theta_mean"] = eos.ptmp(argo.Smean, argo.Tmean, argo.pres)

    return argo


def calc_wind_power_input(
    tau, mld, f0, time_dim="time", r0_factor=0.15, critical_freq_factor=0.5
):
    """
    Solve the Pollard & Millard (1970) slab mixed layer model spectrally
    following Alford (2003).

    Parameters
    ----------

    tau : complex, xarray.DataArray
        1D complex wind stress time series [N/m²].
        Time co-ordinate must be named "time".
    mld : float
        mixed layer depth [m]. Can be time series.
    f0 : float
        inertial frequency, 2Ωsinφ [rad/s].
    time_dim : optional
        Name of time dimension in tau.
    r0_factor : optional, default
        Non-dimensional factor for maximum damping calculated as
            r0 = r0_factor * f0
        Default is 0.15 (Alford, 2003)
    critical_freq_factor : optional, default
        Non-dimensional factor for damping decay scale in frequency
            σc = critical_freq_factor * f0
        Default is 0.5

    Returns
    -------
    windinput : float, xarray.DataArray
        Time series of near-inertial power input [W/m²].
    ZI : complex, xarray.DataArray
        Time series of inertial currents [m/s]

    References
    ----------
    Alford, M.H., 2003. Improved global maps and 54-year history of wind-work
        on ocean inertial motions. Geophys. Res. Lett. 30.

    Pollard, R.T., Millard, R.C., 1970. Comparison between observed and
        simulated wind-generated inertial oscillations.
        Deep Sea Res. Oceanogr. Abstr. 17, 813–821.
    """

    import xrft

    T = tau / 1025
    That = xrft.dft(T, dim=[time_dim], shift=False)
    σ = That["freq_" + time_dim] * 2 * np.pi
    σc = f0 * critical_freq_factor

    # damping
    r = r0_factor * f0 * (1 - np.exp(-0.5 * (σ / σc) ** 2))
    r.name = "damping"

    # transfer functions
    R = (r - 1j * (f0 + σ)) / (r ** 2 + (f0 + σ) ** 2)
    RE = (r - 1j * f0) / (r ** 2 + f0 ** 2)
    RI = R - RE

    # plt.plot(σ / f0, np.real(R), 'k')
    # plt.plot(σ / f0, np.real(RE), 'r')
    # plt.plot(σ / f0, np.real(RI), 'g')
    # plt.gca().set_yscale('log')
    # plt.gca().set_xlim([-3, 1.5])
    # plt.gca().axvline(-1)

    axis = That.get_axis_num("freq_" + time_dim)

    ZI = tau.copy(data=np.fft.ifft(That * RI, axis=axis)) / mld
    # ZE = tau.copy(data=np.fft.ifft(That * RE, axis=axis))
    Z = tau.copy(data=np.fft.ifft(That * R, axis=axis)) / mld

    # windinput_approx = np.real(1025 * ZI * np.conj(T))
    # windinput_approx.attrs['long_name'] = 'Wind power input $Π$'
    # windinput_approx.attrs['units'] = 'W/m$^2$'

    windinput = np.real(1025 * Z * np.conj(T))
    windinput.attrs["long_name"] = "Wind power input $Π$"
    windinput.attrs["units"] = "W/m$^2$"

    ZI.attrs["long_name"] = "Predicted inertial currents"
    ZI.attrs["units"] = "m/s"

    return windinput, ZI


def read_tropflux():

    tx = xr.open_mfdataset(
        "/home/deepak/datasets/tropflux/taux_tropflux_1d_*.nc"
    )  # noqa
    ty = xr.open_mfdataset(
        "/home/deepak/datasets/tropflux/tauy_tropflux_1d_*.nc"
    )  # noqa

    tropflux = xr.merge([tx, ty]).rename({"longitude": "lon", "latitude": "lat"})
    tropflux["tau"] = np.hypot(tropflux.taux, tropflux.tauy)

    return tropflux


def read_oscar(dirname="/home/deepak/work/datasets/oscar/"):
    oscar = (
        xr.open_mfdataset(
            dirname + "/oscar_vel*.nc", drop_variables=["year", "um", "vm"]
        )
        .squeeze()
        .rename({"latitude": "lat", "longitude": "lon"})
        .sortby("lat")
    )

    return oscar


def read_mimoc(dirname="~/datasets/mimoc/", year=2014):

    dirname = os.path.expanduser(dirname)
    mimoc = xr.open_mfdataset(
        f"{dirname}/MIMOC_ML_*.nc", concat_dim="month", combine="nested"
    )
    mimoc["LATITUDE"] = mimoc.LATITUDE.isel(month=1)
    mimoc["LONGITUDE"] = mimoc.LONGITUDE.isel(month=1)
    mimoc = mimoc.swap_dims({"LAT": "LATITUDE", "LONG": "LONGITUDE"}).rename(
        {"LATITUDE": "latitude", "LONGITUDE": "longitude"}
    )
    mimoc["longitude"].attrs.update({"units": "degrees_east", "axis": "X"})
    mimoc["latitude"].attrs.update({"units": "degrees_north", "axis": "Y"})
    mimoc["month"] = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="SM")[::2]
    mimoc = mimoc.rename({"month": "time"})

    return mimoc


def read_oaflux():
    def preprocess(ds):
        year = ds.encoding["source"][-7:-3]
        time = pd.date_range(year + "-01-01", year + "-12-31", freq="D")
        ds["time"].values = time.values

        return ds

    oaflux = xr.open_mfdataset(
        "../datasets/oaflux/evapr_oaflux_201*.nc", preprocess=preprocess
    )
    del oaflux.time.attrs["units"]

    return oaflux


def read_nio():

    nio = xr.open_dataset(
        "~/work/datasets/nio-atlas/"
        + "nioa_climatology_seasonal_temp_salt_monsoon_season.nc",
        decode_times=False,
    ).rename(
        {
            "TIME": "time",
            "DEPTH": "depth",
            "LATITUDE": "lat",
            "LONGITUDE": "lon",
            "SALT": "S",
            "TEMP": "T",
        }
    )

    nio.time.values = ["NE", "NESW", "SW", "SWNE"]

    return nio


def sdif(T):
    """Fickian diffusivity of salt in seawater.
    Input
    -----
    T : float,
        Temperature

    Returns
    -------
    diffusivity

    References
    ----------
    Numerical data and functional relationships in Science and Technology'
    Oceanography v.3. pg 257 -- from Caldwall 1973 and 1974
    """

    # from mixingsoftware/seawater/sw_sdif.m

    return 1e-11 * (62.5 + 3.63 * T)


def tcond(S, T, P):

    ak0 = 0.565403020 + T * (1.6999346e-3 - T * 5.910632e-6)
    f = 0.0690 - 8e-5 * T - 2.0e-7 * P - 1.0e-4 * S
    tcond = ak0 * (1 + f)

    return tcond


def tdif(S, T, P):

    return tcond(S, T, P) / sw.eos80.cp(S, T, P) / sw.eos80.dens(S, T, P)


def visc(S, T, P):
    """
    SW_VISC   kinematic viscosity
    ===========================================================================
    SW_VISC  $Revision: 0.0 $  $Date: 1998/01/19 $
             Copyright (C) Ayal Anis 1998.

    USAGE:  visc = sw_visc(S,T,P)

    DESCRIPTION:
     Calculates kinematic viscosity of sea-water.
     based on Dan Kelley's fit to Knauss's TABLE II-8

    INPUT:  (all must have same dimensions)
    S  = salinity    [psu      (PSS-78) ]
    T  = temperature [degree C (IPTS-68)]
    P  = pressure    [db]
        (P may have dims 1x1, mx1, 1xn or mxn for S(mxn) )

    OUTPUT:
    visc = kinematic viscosity of sea-water [m^2/s]

    visc(40.,40.,1000.)=8.200167608E-7
    """

    visc = (
        1e-4 * (17.91 - 0.5381 * T + 0.00694 * T ** 2 + 0.02305 * S) / sw.dens(S, T, P)
    )

    return visc


def neutral_density(ds):
    """Xarray wrapper for Eric Firing's netural density wrapper"""

    import pygamma

    def gamma_n_wrapper(s, t, p, lon, lat):
        if np.isscalar(lat):
            lat = lat * np.ones_like(s[..., 0])
        if np.isscalar(lon):
            lon = lon * np.ones_like(s[..., 0])
        mask = ~np.isnan(s + t + p)
        g = np.full(s.shape, np.nan)
        g[mask] = pygamma.gamma_n(s[mask], t[mask], p[mask], lon, lat)[0]
        g[g < 0.1] = np.nan  # bad values are 0?
        return g

    lat = ds.cf.coordinates["latitude"][0]
    lon = ds.cf.coordinates["longitude"][0]
    # if ds.cf["sea_water_salinity"].ndim > 2:
    # pygamma only accepts 2D at the most.
    # and depth must be the second dimension
    #     stacked = True
    #     ds = ds.cf.stack(latlon=["latitude", "longitude"]).transpose("latlon", ...)
    # else:
    #     stacked = False

    P = ds.cf["sea_water_pressure"]
    if P.ndim == 1:
        Z = P.dims[0]
    else:
        Z = ds.cf["sea_water_pressure"].cf.axes["Z"][0]

    if "sea_water_salinity" in ds.cf.standard_names:
        S = ds.cf["sea_water_salinity"]
    elif "sea_water_practical_salinity" in ds.cf.standard_names:
        S = ds.cf["sea_water_practical_salinity"]
    else:
        raise ValueError("Could not find practical salinity variable.")

    dtype = S.dtype
    gamma = xr.apply_ufunc(
        gamma_n_wrapper,
        S,
        ds.cf["sea_water_temperature"],
        P,
        ds[lon],
        ds[lat],
        input_core_dims=[[Z], [Z], [Z], [], []],
        output_core_dims=[[Z]],
        dask="parallelized",
        vectorize=True,
        dask_gufunc_kwargs=dict(meta=np.ndarray((0, 0), dtype=dtype)),
        keep_attrs=True,
    )
    gamma.attrs["standard_name"] = "neutral_density"
    gamma.attrs["units"] = "kg/m3"
    gamma.attrs["long_name"] = "$γ_n$"

    return gamma


def read_osu_microstructure_mat(
    fname, coords=("depth", "time"), rename=True, rename_vars=None, no_TS=False
):
    """Read the OSU Ocean Mixing Group microstructure mat files to xarray Dataset.

    Parameters
    ----------
    fname: str
        File name
    coords: Tuple[str]
        List of variables to consider as dimension coordinates
    rename: bool, optional
        Rename variables to "standard names".
    rename_vars: dict, optional
        Dict to rename variables. Useful for nD dimension coordinate variables
        (usually depth)
    no_TS: bool, optional
        If True, skips getting sea_water_temperature and sea_water_potential_temperature
        use with velocity files

    Returns
    -------
    xarray.DataArray
    """
    import warnings

    from scipy.io import loadmat

    if rename_vars is None:
        rename_vars = {}

    mat = loadmat(fname)
    structs = [name for name in mat.keys() if "__" not in name]

    assert len(structs) == 1
    for sname in structs:
        varnames = mat[sname].dtype.names
        print(f"found variables named {varnames!r}")

        ds = xr.Dataset()
        shapes = {mat[sname][coord][0, 0].squeeze().shape[0]: coord for coord in coords}
        print(f"Using coordinate vars with shapes: {shapes}")

        for var in varnames:
            arr = mat[sname][var][0, 0]

            if np.issubdtype(arr.dtype, np.str_) or np.issubdtype(arr.dtype, np.object):
                print(f"Setting {var} as attr.")
                ds.attrs[var] = str(arr.ravel())
                continue

            if arr.dtype.names:
                warnings.warn(
                    f"Skipping substructure {var} with fields {arr.dtype.names}",
                    UserWarning,
                )
                continue

            dims = [shapes[s] for s in arr.shape if s > 1]
            ds[rename_vars.get(var, var)] = (
                dims,
                arr.squeeze(),
                {"coordinates": " ".join(dims)},
            )

    renamer = {
        "jq": "Jq",
        "nsqr": "N2",
        "nsq": "N2",
        "krho": "Krho",
        "sal": "salt",
        "S": "salt",
        "EPS": "eps",
        "U": "u",
        "V": "v",
        "P": "pres",
        "DTDZ": "dTdz",
        "EPSILON": "eps",
        "CHI": "chi",
        "SIGMA": "pden",
        "SIGT": "pden",
        "THETA": "theta",
    }

    if rename:
        if "SIGT" in ds:
            ds["SIGT"] += 1000
        ds = ds.rename({k: v for k, v in renamer.items() if k in ds})

    if "pres" not in ds:
        if "lat" in ds:
            lat = ds.lat.mean()
            lat.attrs.clear()
        else:
            lat = 0
        if "depth" in ds:
            ds["pres"] = eos.pres(ds.depth, lat=lat)

    if not no_TS:
        if "theta" not in ds:
            ds["theta"] = eos.ptmp(ds.salt, ds.T, ds.pres)
        if "T" not in ds:
            ds["T"] = eos.temp(ds.salt, ds.theta, ds.pres)

    attrs = {
        "T": {"standard_name": "sea_water_temperature", "units": "celsius"},
        "theta": {
            "standard_name": "sea_water_potential_temperature",
            "units": "celsius",
        },
        "pres": {"standard_name": "sea_water_pressure", "units": "dbar"},
        "salt": {"standard_name": "sea_water_salinity", "units": "psu"},
        "lon": {"standard_name": "longitude", "units": "degrees_east"},
        "lat": {"standard_name": "latitude", "units": "degrees_north"},
        "eps": {"long_name": "$ε$", "units": "W/kg"},
        "chi": {"long_name": "$χ$", "units": "°C²/s"},
    }
    ds["time"] = util.datenum2datetime(ds.time.data)
    if "depth" in ds:
        ds["depth"].attrs.update({"positive": "down", "axis": "Z"})
        ds["depth"] = ds.depth.astype(float)

    for known in attrs:
        for var in ds.variables:
            if var == known or (known != "T" and var.startswith(known)):
                # deals with things like eps1, eps2
                ds[var].attrs.update(attrs[known])

    return ds


def read_cchdo_chipod_file(file, chunks=None):
    if chunks is None:
        chunks = {"time": 10000}

    chi = (
        xr.open_dataset(file, decode_cf=False, chunks=chunks)
        .swap_dims({"timeSeries": "depth"})
        .rename({"Kt": "KT", "Nsqr": "N2"})
    )
    for var in chi.variables:
        if "FillValue" in chi[var].attrs:
            chi[var].attrs["_FillValue"] = int(chi[var].attrs["FillValue"])
    chi = xr.decode_cf(chi)
    chi["T"] -= 273
    chi.T.attrs["units"] = "C"
    chi = chi.drop_vars(["mooring", "crs", "chipod"])
    return chi


def read_kunze_2017_finestructure(dirname="/home/deepak/datasets/finestructure/"):
    """Reads the Kunze et al (2017) finestructure estimate CSV file"""

    kunze = pd.read_csv(
        f"{dirname}/kunze/strainfineoutatlantic.dat",
        header=None,
        sep=" ",
        index_col=False,
        names=[
            "cruise",
            "drop",
            "latitude",
            "longitude",
            "z_i",
            "z_f",
            "γ_i",
            "γ_f",
            "N2",
            "ε",
            "K",
        ],
    )
    kunze["z_mean"] = (kunze["z_i"] + kunze["z_f"]) / 2
    kunze["γ_mean"] = (kunze["γ_i"] + kunze["γ_f"]) / 2

    return kunze


@numba.guvectorize("(m), (m), () -> (m)", nopython=True)
def sortby(tosort, by, ascending, out):
    mask = ~np.isnan(by)
    idx = np.argsort(by[mask])

    out[:] = np.nan
    out[mask] = tosort[mask][idx]
    if not ascending:
        mask = ~np.isnan(out)
        out[mask] = out[mask][::-1]


def thorpesort(field, by, core_dim="depth", ascending=True):
    """Numba accelerate Thorpe sorting."""

    def wrapper(tosort, by, ascending):
        if tosort.dtype.kind not in "cif":
            return tosort
        out = np.empty_like(tosort)
        sortby(tosort, by, ascending, out)
        return out

    if isinstance(by, str):
        by = field[by]
    return xr.apply_ufunc(
        wrapper,
        field,
        by,
        input_core_dims=[[core_dim], [core_dim]],
        output_core_dims=[[core_dim]],
        dask="parallelized",
        kwargs=dict(ascending=ascending),
    )
