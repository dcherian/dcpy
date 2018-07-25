import seawater as sw
import matplotlib.pyplot as plt
import cmocean as cmo
import numpy as np
import pandas as pd
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
    return 2*(2*π/86400) * np.sin(lat * π/180)


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
            np.reshape(variable[:, :, latind, lonind+1], shape),
            np.reshape(variable[:, :, latind+1, lonind], shape),
            np.reshape(variable[:, :, latind+1, lonind+1], shape)),
                             axis=len(shape)-1)
        avg[avg > 50] = np.nan
        return np.nanmean(avg, axis=len(shape)-1)

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
    omg = np.logspace(np.log10(1.01*f), np.log10(N), 401)

    # horizontal wavenumber
    k = 2*np.pi*np.logspace(-6, -2, 401)

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


def TSplot(S, T, P, Pref=0, ax=None, rho_levels=None,
           label_spines=True, **kwargs):

    # colormap = cmo.cm.matter

    if ax is None:
        plt.figure()
        ax = plt.gca()

    color = kwargs.pop('color', 'teal')
    marker = kwargs.pop('marker', '.')
    fontsize = kwargs.pop('fontsize', 9)

    ax.plot(S, T, ls='None', marker=marker, color=color, **kwargs)
    # ax.scatter(S, T, s=4*120, c=P,
    #            alpha=0.5, linewidth=0.15, edgecolor='gray',
    #            cmap=colormap, zorder=-10)
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

    cs = ax.contour(Smat, Tmat, ρ, colors='gray',
                    levels=rho_levels, linestyles='solid')

    if label_spines:
        # needed to do some labelling setup
        clabels = ax.clabel(cs, fmt='%.1f', inline=False)

        [txt.set_visible(False) for txt in clabels]

        for idx, _ in enumerate(cs.levels):
            # This is the rightmost point on each calculated contour
            x, y = cs.allsegs[idx][0][0, :]
            # This is a very helpful function!
            cs.add_label_near(x, y, inline=False, inline_spacing=0)

        xlim = ax.get_xlim()

        def edit_text(t):
            if abs(t.get_position()[0] - xlim[1])/xlim[1] < 1e-6:
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
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)

    else:
        clabels = ax.clabel(cs, fmt='%.1f', inline=True, inline_spacing=10)
        [txt.set_backgroundcolor([0.95, 0.95, 0.95, 0.75]) for txt in clabels]

    ax.text(0, 0.995, ' $σ_' + str(Pref) + '$', transform=ax.transAxes,
            va='top', fontsize=fontsize+2, color='gray')

    ax.set_xlabel('S')
    ax.set_ylabel('T')

    return cs


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

    aq = xr.open_mfdataset(dirname+'/*.nc',
                           autoclose=True,
                           preprocess=preprocess)

    aq['latitude'].attrs['units'] = 'degrees_east'
    aq['longitude'].attrs['units'] = 'degrees_north'
    aq.sss.attrs['long_name'] = 'Sea Surface Salinity'

    aq = aq.sortby('latitude')

    return aq
