#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import datetime as dtm
from sqlalchemy import create_engine
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely import geometry
import xarray as xr
import netCDF4 as nc4
from loguru import logger

import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def runOsmium(input_pbf, output='output_osmium'):
    '''
    Runs Osmium to extract the data from OSM
    '''
    osmium = os.system(f'osmium tags-filter -v -o {output}.osm.pbf {input_pbf} w/highway r/boundary=administrative')

    # cmd = ['osmium', '--write-traject', '--state-file', self.commfile.filepath]
    
    logger.info('Osmium run')

    return osmium

def createDatabase(user, password, host, port, database):
    '''
    Creates a database with postgis extension
    '''
    # Creating the database
    engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/postgres')
    connection = engine.connect()
    connection.execution_options(isolation_level="AUTOCOMMIT").execute(f'CREATE DATABASE {database};')
    connection.close()

    # Creating the postgis extension
    engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{database}')
    connection = engine.connect()
    connection.execution_options(isolation_level="AUTOCOMMIT").execute(f'CREATE EXTENSION postgis;')
    connection.execution_options(isolation_level="AUTOCOMMIT").execute(f'CREATE EXTENSION hstore;')
    connection.close()

    logger.info('Database created')

    return engine

def runOsm2pgsql(database, lua_file='highways.lua', input='output_osmium'):
    '''
    Runs Osm2pgsql to extract the data from OSM
    '''
    osm2pgsql = os.system(f'osm2pgsql -d {database} -O flex -S {lua_file} {input}.osm.pbf')

    logger.info('Osm2pgsql run')

    return osm2pgsql

def createEngine(user, password, host, port, database):
    '''
    Creates a connection to a database
    '''
    engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{database}')

    logger.info('Connection to database created')

    return engine

def drawGrid(geom, square_size, proj=None):
    '''
    Creates a grid with squared grid cells around a defined geometry in geografical coordinates or projected if specified.
    square_size: size of the cell in degrees
    '''
    total_bounds = geom.total_bounds
    minX, minY, maxX, maxY = total_bounds

    minX = round(minX, 1)
    minY = round(minY, 1)

    if maxX - round(maxX, 1) <= square_size:
        maxX = round(maxX, 1)
    else:
        maxX = round(maxX, 1) + square_size

    if maxY - round(maxY, 1) <= square_size:
        maxY = round(maxY, 1)
    else:
        maxY = round(maxY, 1) + square_size
    
    x, y = (minX, minY)
    geom_array = []

    while y <= maxY-square_size:
        while x <= maxX-square_size:
            geom = geometry.Polygon([(x,y), (x, y+square_size), (x+square_size, y+square_size), (x+square_size, y), (x, y)])
            geom_array.append(geom)
            x += square_size
        x = minX
        y += square_size

    grid = gpd.GeoDataFrame(geom_array, columns=['geometry'])
    grid = grid.set_crs('EPSG:4326')

    if proj is not None:
        grid = grid.to_crs(proj)

    logger.info(f'Grid created with {grid.shape[0]} cells')

    return grid

def readData(engine, weights=False, proj=None):
    '''
    Reads data from a database table and returns a geopandas dataframe
    '''
    data = gpd.read_postgis("SELECT * FROM highways WHERE type = 'motorway' OR type = 'trunk' OR type = 'primary' OR type = 'secondary' OR type = 'tertiary' OR type = 'motorway_link' OR type = 'trunk_link' OR type = 'primary_link' OR type = 'secondary_link' OR type = 'tertiary_link'", engine, geom_col='geom')

    if weights:
        boundaries = gpd.read_postgis("SELECT * FROM boundaries", engine, geom_col='geom') # Read administrative boundaries
        data = data.sjoin(boundaries, how='left') # Spatial join with administrative boundaries
        data = data.drop(columns=['index_right']) # Drop index column
        data.lanes = data.lanes.fillna(data.groupby(['name', 'type']).lanes.transform('median')) # Fill NaN values with median of the same road type and administrative boundary
        data.lanes = data.lanes.fillna(data.groupby(['type']).lanes.transform('median'))  # Fill NaN values with median of the same road type (just in case a road type is not present in a specific administrative boundary)
        data.lanes = data.lanes.fillna(1) # Fill NaN values with 1 (just in case a road type is not present in a specific administrative boundary (quite unlikely))

    if proj is not None:
        data = data.to_crs(proj)

    logger.info(f'Data read from database {engine}')

    return data

def calculateLengths(data, grid, proj=None, weights=False):
    '''
    Calculates the length of the roads in each grid cell
    '''
    grid.loc[:,'cell_idx'] = np.arange(0,grid.shape[0]) # Preparing grid to spatial join with highways

    if grid.crs.is_geographic:
        if proj is not None:
            grid_proj = grid.to_crs(proj)
        else:
            grid_proj = grid.to_crs('EPSG:3857')
            logger.warning('Grid is in geographic coordinates, but no projection is specified. Using EPSG:3857')
    
    if proj is not None:
        data = data.to_crs(proj)
    else:
        data = data.to_crs('EPSG:3857')
        logger.warning('Data is in geographic coordinates, but no projection is specified. Using EPSG:3857')
    
    data.loc[:,'lengths'] = data.length

    # Spatial join
    data = data.sjoin(grid_proj, how='left')

    if weights:
        data.loc[:,'lengths'] = data.lengths * data.lanes

    # Calculate the length of the roads by road type in each grid cell
    data_grid = data.groupby(['type','cell_idx']).sum().reset_index()

    grids = {}

    for rtype in data_grid['type'].unique():
        grids[rtype] = grid.merge(data_grid.loc[data_grid['type'] == rtype], on='cell_idx', how='left')
        grids[rtype].loc[:,'lengths'] = grids[rtype].lengths.fillna(0)

    # Merge the grid with the length of the roads
    data_grid = data.groupby('cell_idx').sum().reset_index()

    total = pd.merge(grid, data_grid, on='cell_idx', how='left')

    grids['total'] = total
    grids['total'].loc[:,'lengths'] = grids['total'].lengths.fillna(0)

    logger.info('Lengths calculated')
    
    return grids

def tableToXarray(grid, filename=None):
    '''
    Converts a geopandas dataframe to an xarray dataset
    '''
    
    # Create an array with the centroids of the grid cells
    centroids = np.array([np.array((round(cell.centroid.x,2),round(cell.centroid.y,2))) for cell in grid['total'].geometry])

    # Extract longitude and latitude from the centroids
    lon = np.unique(centroids[:,0])
    lat = np.unique(centroids[:,1])

    # Create a dictionary with the length of the roads in each grid cell
    lengths = {}
    encoding_dct = {}
    for rtype in grid.keys():
        lengths[rtype] = {"dims": ['lat', 'lon'], 'data': grid[rtype].lengths.values.reshape(len(lat), len(lon)), 'attrs': {'units': 'm'}}
        encoding_dct[rtype] = {"zlib": True, "complevel": 6}
    # Create xarray dataset
    ds = xr.Dataset.from_dict(lengths)
    ds = ds.assign_coords(lon=lon, lat=lat)
    ds.attrs['title'] = 'Road length'
    ds.attrs['units'] = 'm'
    ds.attrs['projection'] = 'EPSG:4326'
    ds.attrs['history']    = ' '.join(sys.argv)
    ds.attrs['date_created'] = dtm.datetime.utcnow().isoformat()
    logger.info('Xarray dataset created')

    if filename is not None:
        ds.to_netcdf(filename,
                     engine="netcdf4", encoding=encoding_dct)
        logger.info(f'Xarray dataset stored to {filename}')

    return ds

def plotLengths(ds):
    '''
    Plots the road length in a map
    '''

    # Extract bounds of the dataset
    lon_min = ds.lon.min().values
    lon_max = ds.lon.max().values
    lat_min = ds.lat.min().values
    lat_max = ds.lat.max().values

    # Extent of the map
    extent = [lon_min, lon_max, lat_min, lat_max]

    rtypes = list(ds.data_vars)
    p_array = np.sqrt(len(rtypes))

    if p_array - int(p_array) > 0:
        x = int(p_array) + 1
        y = int(p_array)
    else:
        x = int(p_array)
        y = int(p_array)

    fig, ax = plt.subplots(x, y, figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    
    ax = ax.reshape(-1)
    for i, rtype in enumerate(rtypes):
        ax[i].set_extent(extent, crs=ccrs.PlateCarree())
        ax[i].add_feature(cartopy.feature.BORDERS, linestyle=':')
        ax[i].add_feature(cartopy.feature.LAND)
        ax[i].add_feature(cartopy.feature.OCEAN)
        ax[i].add_feature(cartopy.feature.COASTLINE)
        ax[i].gridlines(draw_labels=True)
        ax[i].xaxis.set_major_formatter(LONGITUDE_FORMATTER)
        ax[i].yaxis.set_major_formatter(LATITUDE_FORMATTER)
        ds[rtype].plot(ax=ax[i], transform=ccrs.PlateCarree(), cmap='viridis', cbar_kwargs={'label': 'Road length (m)'})
        ax[i].set_title(f'Road length ({rtype})')

    fig.tight_layout()

    # Save figure
    fig.savefig('road_length.png', dpi=300, bbox_inches='tight')

    logger.info('Figure saved')

if __name__ == '__main__':

    import sys
    from argparse import ArgumentParser, REMAINDER

    parser = ArgumentParser(description='Create a grid with the length of the roads in each grid cell')
    parser.add_argument('args', nargs=REMAINDER, help='Arguments to pass to the script')
    parser.add_argument('-u', '--user', dest='user', help='Database user (default: %(default)s).', default='postgres')
    parser.add_argument('-p', '--password', dest='password', help='Database password', default='postgres')
    parser.add_argument('-d', '--database', dest='database', help='Database name (default: %(default)s)', default='osmpg_germany')
    parser.add_argument('-s', '--host', dest='host', help='Database host', default='localhost')
    parser.add_argument('-t', '--port', dest='port', help='Database port (default: %(default)s)', default='5432')
    parser.add_argument('-db', '--create-db', dest='create_db', help='Create database', default=False)
    parser.add_argument('-o', '--output', dest='output', help='Output file', default='road_length.nc')
    parser.add_argument('-g', '--grid', dest='grid', help='Grid size in degrees (default: %(default)s)', default=0.1)
    parser.add_argument('-c', '--crs', dest='crs', help='Projection of the grid', default='EPSG:3857')
    parser.add_argument('-i', '--pbf', dest='pbf', help='Input *.osm.pbf file')#, required=True)
    parser.add_argument('-a', '--lua', dest='lua', help='*.lua file')#, required=True)
    parser.add_argument('-w', '--weights', dest='weights', help='Weight the road lengths by number of lanes', action='store_true')
    parser.add_argument('-l', '--log', dest='log', help='Log file', default='log.txt')
    parser.add_argument('-v', '--verbose', dest='verbose', help='Verbose', action='store_true')
    parser.add_argument('-q', '--quiet', dest='quiet', help='Quiet', action='store_true')
    parser.add_argument('-f', '--force', dest='force', help='Force', action='store_true')
    parser.add_argument('-n', '--no-plot', dest='plot', help='Do not plot', action='store_false')
    parser.add_argument('-m', '--map', dest='map', help='Plot map', action='store_true')

    args = parser.parse_args(sys.argv[1:])

    # Set the verbosity in the logger (loguru quirks ...)
    logger.remove()
    logger.add(sys.stderr, level=args.verbose)

    # Run osmium
    if args.pbf:
        logger.info("extracting infomartion from -->{}<--...".format(args.pbf))
        runOsmium(args.pbf)
        logger.info("...information extracted from -->{}<--.".format(args.pbf))
    else:
        logger.info("no *.osm.pbf file given, skipping osmium")

    # Create database
    if args.create_db:
        logger.info("creating database -->{}<--...".format(args.database))
        createDatabase(args.user, args.password, args.host, args.port, args.database)
        logger.info("...database -->{}<-- created.".format(args.database))
    else:
        logger.info("no database creation requested, skipping database creation")

    # Run osm2pgsql
    if args.lua:
        logger.info("loading information to database -->{}<--...".format(args.database))
        runOsm2pgsql(args.database, args.lua)
        logger.info("...information loaded to -->{}<--.".format(args.database))
    else:
        logger.info("no *.lua file given, skipping osm2pgsql")

    # Create connection to database
    logger.info("connecting to database -->{}<--...".format(args.database))
    engine = createEngine(args.user, args.password, args.host, args.port, args.database)
    logger.info("...database -->{}<-- connected.".format(args.database))
    
    # Read data from database
    assert args.weights==False, \
        "osm2pgsql databases were not yet created with support for administrative boundaries!"
    logger.info("reading database table...")
    data = readData(engine, args.weights)
    logger.info("...data read")

    # Create grid
    logger.info("start grid preparation...")
    grid = drawGrid(data, float(args.grid))
    logger.info("...grid prepared.")

    # Calculate lengths
    logger.info("start road length computation...")
    grid = calculateLengths(data, grid, args.crs, args.weights)
    logger.info("...road length done.")

    # Create xarray dataset
    logger.info("start output generation...")
    ds = tableToXarray(grid, args.output)
    logger.info("...generated file ***{}***".format(args.output))

    # Save map
    if args.map:
        logger.info("start map generation...")
        plotLengths(ds)
        logger.info("...map generated.")