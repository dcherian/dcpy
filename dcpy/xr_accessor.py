import xarray as xr


@xr.register_dataarray_accessor("dc")
class DcAccessor(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def set_name_units(self, name, units):
        self._obj.attrs["long_name"] = name
        self._obj.attrs["units"] = units
