import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seawater as sw

import xarray as xr


def dataset_center_pacific(da, name=None):
    ''' Takes an input DataArray and rolls the longitude
        so that all 3 basins are covered. Longitude goes from 20 to 380. '''

    if name is None:
        name = 'longitude' if 'longitude' in da.coords else 'lon'

    # roll so that pacific is in the middle
    # and we have coverage of all 3 basins

    da = da.roll(**{name: -1 * da[name].searchsorted(20)})

    coord = da[name].values
    coord[coord < 20] += 360

    da[name].values = coord

    return da


def coriolis(lat):
    π = np.pi
    return 2 * (2 * π / 86400) * np.sin(lat * π / 180)


def ReadWoa(lon, lat, time='annual', depth=None, return_xr=False):
    ''' Given lon, lat and type, return WOA data.
        Input:
              lon : +ve East
              lat : +ve North
              time: 'annual' or 'seasonal'
              depth: float (m)
        Output:
              Returns a dictionary with T, S, depth
    '''
    import numpy as np
    import netCDF4 as nc

    woa = dict()

    # read in World Ocean Atlas data
    if time == 'annual':
        woaT = nc.Dataset('/home/deepak/datasets/woa13_decav_t00_01v2.nc', 'r')
        woaS = nc.Dataset('/home/deepak/datasets/woa13_decav_s00_01v2.nc', 'r')

    if time == 'seasonal':
        woaT = nc.MFDataset('/home/deepak/datasets/woa13-season/'
                            + 'woa13_decav_t*_01v2.nc', 'r', aggdim='time')
        woaS = nc.MFDataset('/home/deepak/datasets/woa13-season/'
                            + 'woa13_decav_s*_01v2.nc', 'r', aggdim='time')

    latind = np.where(woaT['lat'][:] < lat)[0][-1]
    lonind = np.where(woaT['lon'][:] < lon)[0][-1]

    def ConcatAndAverage(variable, lonind, latind):
        shape = list(variable[:, :, latind, lonind].shape)
        shape.append(1)

        avg = np.concatenate((
            np.reshape(variable[:, :, latind, lonind], shape),
            np.reshape(variable[:, :, latind, lonind + 1], shape),
            np.reshape(variable[:, :, latind + 1, lonind], shape),
            np.reshape(variable[:, :, latind + 1, lonind + 1], shape)),
            axis=len(shape) - 1)
        avg[avg > 50] = np.nan
        return np.nanmean(avg, axis=len(shape) - 1)

    woa['T'] = ConcatAndAverage(woaT['t_an'], lonind, latind)
    woa['S'] = ConcatAndAverage(woaS['s_an'], lonind, latind)
    woa['depth'] = woaT['depth'][:]

    if depth is not None:
        index = np.where(woaT['depth'][:] == np.abs(depth))
        woa['depth'] = abs(depth)
        woa['T'] = woa['T'][:, index]
        woa['S'] = woa['S'][:, index]

    if return_xr:
        import xarray as xr
        woadict = woa

        woa = xr.Dataset({'T': (['depth'], np.squeeze(woadict['T'])),
                          'S': (['depth'], np.squeeze(woadict['S']))},
                         coords={'depth': (['depth'], woadict['depth'])})
    return woa


def GM(lat, N, N0, b=1000, oned=False):
    try:
        import GM81.gm as gm
    except ImportError:
        raise ImportError('Please install the GM81 package.')

    import seawater as sw
    import numpy as np

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


def TSplot(S, T, P, Pref=0, ax=None, rho_levels=[],
           labels=True, label_spines=True,
           plot_distrib=True, Sbins=30, Tbins=30, **kwargs):

    # colormap = cmo.cm.matter

    defaults = {'edgecolor': 'gray',
                'linewidth': 0.15,
                'alpha': 0.5}

    size = kwargs.pop('size', 32)
    color = kwargs.pop('color', 'teal')
    # marker = kwargs.pop('marker', '.')
    fontsize = kwargs.pop('fontsize', 9)
    defaults.update(kwargs)

    labels = False if rho_levels is None else labels

    def flatten_data(data):
        if isinstance(data, xr.DataArray):
            d = data.values.ravel()
        elif isinstance(data, np.ndarray):
            d = data.ravel()
        else:
            return data

        return d[~np.isnan(d)]

    axes = dict()

    if ax is None:
        f = plt.figure(constrained_layout=True)
        if plot_distrib:
            gs = mpl.gridspec.GridSpec(3, 3, figure=f)

            axes['ts'] = f.add_subplot(gs[1:, :-1])
            axes['s'] = f.add_subplot(gs[0, :-1], sharex=axes['ts'])
            axes['t'] = f.add_subplot(gs[1:, -1], sharey=axes['ts'])
            ax = axes['ts']
        else:
            ax = plt.gca()
    elif isinstance(ax, dict):
        axes = ax
        ax = axes['ts']

    axes['ts'] = ax

    salt = flatten_data(S)
    temp = flatten_data(T)

    #     ts = ax.plot(S, T, ls='None', marker=marker, color=color, **kwargs)
    ts = ax.scatter(salt, temp, s=flatten_data(size), c=flatten_data(color),
                    **defaults)
    # ts = ax.hexbin(flatten_data(S), flatten_data(T), **defaults)

    # defaults.pop('alpha')
    # ts = ax.scatter(flatten_data(S), flatten_data(T),
    #                 s=flatten_data(size), c=[[0, 0, 0, 0]],
    #                 **defaults)

    Slim = ax.get_xlim()
    Tlim = ax.get_ylim()

    Tvec = np.arange(Tlim[0], Tlim[1], 0.1)
    Svec = np.arange(Slim[0], Slim[1], 0.1)
    [Smat, Tmat] = np.meshgrid(Svec, Tvec)

    ρ = sw.pden(Smat, Tmat, Pref) - 1000

    if rho_levels is not None:
        rho_levels = np.asarray(rho_levels)
        if np.all(rho_levels > 1000):
            rho_levels -= 1000
        if not (rho_levels.size > 0):
            rho_levels = 7

        cs = ax.contour(Smat, Tmat, ρ, colors='gray',
                        levels=rho_levels, linestyles='solid',
                        zorder=-1)

        ax.set_xlim([Smat.min(), Smat.max()])
        ax.set_ylim([Tmat.min(), Tmat.max()])
    else:
        cs = None

    if plot_distrib:
        hist_args = dict(color=color, density=True, histtype='step')
        Thist = axes['t'].hist(temp, orientation='horizontal',
                               bins=Tbins, **hist_args)
        axes['t'].set_xticklabels([])
        axes['t'].set_xticks([])
        axes['t'].spines['bottom'].set_visible(False)

        Shist = axes['s'].hist(salt, bins=Sbins, **hist_args)
        axes['s'].set_yticks([])
        axes['s'].set_yticklabels([])
        axes['s'].spines['left'].set_visible(False)
    else:
        Thist = None
        Shist = None

    if labels:
        if label_spines:
            # needed to do some labelling setup
            clabels = ax.clabel(cs, fmt='%.1f', inline=False)

            [txt.set_visible(False) for txt in clabels]

            for idx, _ in enumerate(cs.levels):
                if not cs.allsegs[idx]:
                    continue

                # This is the rightmost point on each calculated contour
                x, y = cs.allsegs[idx][0][0, :]
                # This is a very helpful function!
                cs.add_label_near(x, y, inline=False, inline_spacing=0)

            xlim = ax.get_xlim()

            def edit_text(t):
                if abs(t.get_position()[0] - xlim[1]) / xlim[1] < 1e-6:
                    # right spine lables
                    t.set_verticalalignment('center')
                else:
                    # top spine labels need to be aligned to the bottom
                    t.set_verticalalignment('bottom')

                t.set_clip_on(False)
                t.set_horizontalalignment('left')
                t.set_size(fontsize)
                t.set_text(' ' + t.get_text())

            [edit_text(t) for t in cs.labelTexts]

        else:
            clabels = ax.clabel(cs, fmt='%.1f', inline=True, inline_spacing=10)
            [txt.set_backgroundcolor([0.95, 0.95, 0.95, 0.75])
             for txt in clabels]

    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)

    ax.text(0, 0.995, ' $σ_' + str(Pref) + '$', transform=ax.transAxes,
            va='top', fontsize=fontsize + 2, color='gray')

    ax.set_xlabel('S')
    ax.set_ylabel('T')

    return cs, ts, axes, Thist, Shist


def argo_mld_clim(kind='monthly', fname=None):
    if fname is None:
        if kind is 'monthly':
            fname = '~/datasets/argomld/Argo_mixedlayers_monthlyclim_03192017.nc'

        if kind is 'annual':
            fname = '~/datasets/argomld/Argo_mixedlayers_all_03192017.nc'

    import xarray as xr

    ds = xr.open_dataset(fname, autoclose=True)

    mld = xr.Dataset()
    month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for da in ds:
        if da[0:2] == 'ml':
            mld[da] = xr.DataArray(ds[da].values,
                                   coords=[('lat', ds['lat']),
                                           ('lon', ds['lon']),
                                           ('month', ds['month'])])

    return mld


def read_trmm():
    def preprocess(ds):
        ds['time'] = pd.to_datetime(ds.attrs['BeginDate']
                                    + ' '
                                    + ds.attrs['BeginTime'])
        ds = ds.expand_dims('time')

        return ds

    trmm = (xr.open_mfdataset('../datasets/trmm/3B42_Daily.*.nc4.nc4',
                              preprocess=preprocess,
                              concat_dim='time',
                              autoclose=True))

    return trmm


def read_aquarius(dirname='/home/deepak/datasets/aquarius/oisss/'):

    import cftime

    def preprocess(ds):
        t = (np.datetime64(cftime.num2date(ds.time[0],
                                           'days since 2010-12-31',
                                           calendar='julian'))
             + np.timedelta64(3, 'D'))

        dsnew = ds.squeeze().drop('time')
        dsnew['time'] = xr.DataArray([t], dims=['time'])
        dsnew['sss'] = dsnew.sss.expand_dims('time')

        return dsnew

    aq = xr.open_mfdataset(dirname + '*.nc', preprocess=preprocess,
                           decode_times=False, autoclose=True)

    return aq


def read_aquarius_l3(dirname='/home/deepak/datasets/aquarius/L3/combined/'):

    def preprocess(ds):

        dsnew = ds.copy()
        dsnew['latitude'] = xr.DataArray(np.linspace(90, -90, 180),
                                         dims=['phony_dim_0'])
        dsnew['longitude'] = xr.DataArray(np.linspace(-180, 180, 360),
                                          dims=['phony_dim_1'])
        dsnew = (dsnew.rename({'l3m_data': 'sss',
                               'phony_dim_0': 'latitude',
                               'phony_dim_1': 'longitude'})
                 .set_coords(['latitude', 'longitude'])
                 .drop('palette'))

        dsnew['time'] = (pd.to_datetime(dsnew.attrs['time_coverage_start'])
                         + np.timedelta64(3, 'D') + np.timedelta64(12, 'h'))
        dsnew = dsnew.expand_dims('time').set_coords('time')

        return dsnew

    aq = xr.open_mfdataset(dirname + '/*.nc',
                           autoclose=True,
                           preprocess=preprocess)

    aq['latitude'].attrs['units'] = 'degrees_east'
    aq['longitude'].attrs['units'] = 'degrees_north'
    aq.sss.attrs['long_name'] = 'Sea Surface Salinity'

    aq = aq.sortby('latitude')

    return aq


def read_argo_clim(dirname='/home/deepak/datasets/argoclim/'):

    chunks = {'LATITUDE': 10, 'LONGITUDE': 10}

    argoT = xr.open_dataset(dirname + 'RG_ArgoClim_Temperature_2016.nc',
                            decode_times=False, chunks=chunks)
    argoS = xr.open_dataset(dirname + 'RG_ArgoClim_Salinity_2016.nc',
                            decode_times=False, chunks=chunks)

    argoS['S'] = argoS.ARGO_SALINITY_ANOMALY + argoS.ARGO_SALINITY_MEAN
    argoT['T'] = argoT.ARGO_TEMPERATURE_ANOMALY + argoT.ARGO_TEMPERATURE_MEAN

    argo = (xr.merge([argoT, argoS]))

    argo = (argo.rename({'LATITUDE': 'lat',
                         'LONGITUDE': 'lon',
                         'PRESSURE': 'pres',
                         'TIME': 'time',
                         'ARGO_TEMPERATURE_MEAN': 'Tmean',
                         'ARGO_TEMPERATURE_ANOMALY': 'Tanom',
                         'ARGO_SALINITY_MEAN': 'Smean',
                         'ARGO_SALINITY_ANOMALY': 'Sanom'}))

    _, ref_date = xr.coding.times._unpack_netcdf_time_units(argo.time.units)

    argo.time.values = (pd.Timestamp(ref_date)
                        + pd.to_timedelta(30 * argo.time, unit='D'))

    return argo


def calc_wind_power_input(tau, mld, f0, time_dim='time'):
    """
    Solve the Pollard & Millard (1970) slab mixed layer model spectrally
    following Alford (2003).

    Parameters
    ----------

    tau : complex, xarray.DataArray
        1D complex wind stress time series [N/m²].
        Time co-ordinate must be named "time".
    mld : float
        mixed layer depth [m].
    f0 : float
        inertial frequency [cycles per seconds].
    time_dim: optional
        Name of time dimension in tau.

    Returns
    -------
    windinput : float, xarray.DataArray
        Time series of near-inertial power input in W/m².

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
    That = xrft.dft(tau, dim=[time_dim], shift=False)
    σ = That['freq_' + time_dim]
    σc = f0 / 2

    # damping
    r = 0.15 * f0 * (1 - np.exp(-0.5 * (σ / σc)**2))
    r.name = 'damping'

    # transfer functions
    R = 1 / mld * (r - 1j * (f0 + σ)) / (r**2 + (f0 + σ)**2)
    RE = 1 / mld * (r - 1j * f0) / (r**2 + f0**2)
    RI = R - RE

    # plt.plot(σ / f0, np.real(R), 'k')
    # plt.plot(σ / f0, np.real(RE), 'r')
    # plt.plot(σ / f0, np.real(RI), 'g')
    # plt.gca().set_yscale('log')
    # plt.gca().set_xlim([-3, 1.5])
    # plt.gca().axvline(-1)

    ZI = tau.copy(data=np.fft.ifft(That * RI,
                                   axis=That.get_axis_num('freq_' + time_dim)))
    # ZE = tau.copy(data=np.fft.ifft(That * RE,
    #                                axis=That.get_axis_num('freq_' + time_dim)))
    # Z = tau.copy(data=np.fft.ifft(That * R,
    #                               axis=That.get_axis_num('freq_' + time_dim)))

    windinput = np.real(1025 * ZI * np.conj(T))
    windinput.attrs['long_name'] = 'Wind power input $Π$'
    windinput.attrs['units'] = 'W/m$^2$'

    return windinput
