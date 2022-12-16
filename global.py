#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xarray as xr
import numpy as np
import os 

def mergeContinents(square_size, continents, prefix):

    lon = np.arange(-180+square_size/2, 180, square_size)
    lat = np.arange(-90+square_size/2, 90, square_size)

    for i, fl in enumerate(continents):
        ds = xr.open_dataset(prefix + fl)
        ds = ds.interp(lon=lon, lat=lat)
        ds = ds.fillna(0)
        ds = ds.total
        if i == 0:
            ds_out = ds.copy()
        else:
            ds_out += ds

    ds_out.to_netcdf('continents.nc')


if __name__ == '__main__':

    square_size = 0.1
    prefix = '../continents/'
    continents = os.listdir('../continents/')
    mergeContinents(square_size, continents, prefix)
