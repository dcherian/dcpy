import seawater as sw
import matplotlib.pyplot as plt
import cmocean as cmo
import numpy as np


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


def TSplot(S, T, P, Pref=0, ax=None):

    colormap = cmo.cm.matter

    if ax is None:
        plt.figure()
        ax = plt.gca()

    ax.plot(S, T, '.', color='teal')
    # ax.scatter(S, T, s=4*120, c=P,
    #            alpha=0.5, linewidth=0.15, edgecolor='gray',
    #            cmap=colormap, zorder=-10)
    Slim = ax.get_xlim()
    Tlim = ax.get_ylim()

    Tvec = np.arange(Tlim[0], Tlim[1], 0.1)
    Svec = np.arange(Slim[0], Slim[1], 0.1)
    [Smat, Tmat] = np.meshgrid(Svec, Tvec)

    ρ = sw.pden(Smat, Tmat, Pref) - 1000

    cs = ax.contour(Smat, Tmat, ρ, colors='gray',
                    linestyles='dashed')
    ax.clabel(cs, fmt='%.1f')

    ax.set_xlabel('S')
    ax.set_ylabel('T')


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
