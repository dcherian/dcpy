import seawater as sw
import matplotlib.pyplot as plt
import cmocean as cmo
import numpy as np


def inertial(lat):
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
