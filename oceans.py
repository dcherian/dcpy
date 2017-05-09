def ReadWoa(lon, lat, time='annual', depth=None):
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

    latind = np.where(woaT['lat'][:] < 12)[0][-1]
    lonind = np.where(woaT['lon'][:] < 90)[0][-1]

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

    return woa
