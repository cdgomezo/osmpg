from sqlalchemy import create_engine
from pandas import merge
from geopandas import read_postgis, GeoDataFrame
from numpy import arange, array, unique
from shapely import geometry
from xarray import Dataset
from loguru import logger

import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


# Function to create grid
def drawGrid(geom, square_size, proj=None):
    '''
    Creates a grid with squared grid cells around a defined geometry in geografical coordinates or projected if specified.
    square_size: size of the cell in degrees
    '''
    total_bounds = geom.total_bounds
    minX, minY, maxX, maxY = total_bounds

    minX = int(minX)
    minY = int(minY)

    maxX = int(maxX * 10) / 10
    
    if maxX/int(maxX) != 1:
        maxX = maxX + square_size
    
    maxY = int(maxY * 10) / 10

    if maxY/int(maxY) != 1:
        maxY = maxY + square_size
    
    x, y = (minX, minY)
    geom_array = []

    while y <= maxY-square_size:
        while x <= maxX:
            geom = geometry.Polygon([(x,y), (x, y+square_size), (x+square_size, y+square_size), (x+square_size, y), (x, y)])
            geom_array.append(geom)
            x += square_size
        x = minX
        y += square_size

    grid = GeoDataFrame(geom_array, columns=['geometry'])
    grid = grid.set_crs('EPSG:4326')

    if proj is not None:
        grid = grid.to_crs(proj)

    logger.info(f'Grid created with {grid.shape[0]} cells')

    return grid

# Function to create grid
def createEngine(user, password, host, port, database):
    '''
    Creates a connection to a database
    '''
    engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{database}')

    logger.info('Connection to database created')

    return engine

# Function to read the data
def readData(engine, proj=None):
    '''
    Reads data from a database table and returns a geopandas dataframe
    '''
    data = read_postgis("SELECT * FROM highways WHERE type = 'motorway' OR type = 'trunk' OR type = 'primary' OR type = 'secondary' OR type = 'tertiary' OR type = 'motorway_link' OR type = 'trunk_link' OR type = 'primary_link' OR type = 'secondary_link' OR type = 'tertiary_link'", engine, geom_col='geom')
    if proj is not None:
        data = data.to_crs(proj)

    logger.info(f'Data read from database {engine}')

    return data

# Function to calculate lengths
def calculateLengths(data, grid, proj=None):
    '''
    Calculates the length of the roads in each grid cell
    '''
    grid.loc[:,'cell_idx'] = arange(0,grid.shape[0]) # Preparing grid to spatial join with highways

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
    
    data.loc[:,'length'] = data.length

    # Spatial join
    data = data.sjoin(grid_proj, how='left')

    # Calculate the length of the roads in each grid cell
    data_grid = data.groupby('cell_idx').sum().reset_index()

    # Merge the grid with the length of the roads
    grid = merge(grid, data_grid, on='cell_idx', how='left')

    logger.info('Lengths calculated')
    
    return grid

def tableToXarray(grid, filename=None):
    '''
    Converts a geopandas dataframe to an xarray dataset
    '''
    
    # Create an array with the centroids of the grid cells
    centroids = array([array((round(cell.centroid.x,2),round(cell.centroid.y,2))) for cell in grid.geometry])

    # Extract longitude and latitude from the centroids
    lon = unique(centroids[:,0])
    lat = unique(centroids[:,1])

    # Create matrix with the length of the roads in each grid cell
    road_length = array(grid['length'].values).reshape(len(lat), len(lon))

    # Create xarray dataset
    ds = Dataset({'road_length': (['lat', 'lon'], road_length[:,:])}, coords={'lat': lat, 'lon': lon})
    ds.attrs['title'] = 'Road length'
    # ds.attrs['description'] = 'Road length in Germany'
    ds.attrs['units'] = 'm'
    ds.attrs['projection'] = 'EPSG:4326'
    # ds.attrs['year'] = 2019

    logger.info('Xarray dataset created')

    if filename is not None:
        ds.to_netcdf(filename)
        logger.info(f'Xarray dataset stored in {filename}')

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

    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
    ax.add_feature(cartopy.feature.LAND)
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.COASTLINE)
    ax.gridlines(draw_labels=True)
    ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
    ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)
    ax.set_title('Road length')
    ds.road_length.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='viridis', cbar_kwargs={'label': 'Road length (m)'})

    # Save figure
    fig.savefig('road_length.png', dpi=300, bbox_inches='tight')

    logger.info('Figure saved')

if __name__ == '__main__':

    import sys
    from argparse import ArgumentParser, REMAINDER

    parser = ArgumentParser(description='Create a grid with the length of the roads in each grid cell')
    parser.add_argument('args', nargs=REMAINDER, help='Arguments to pass to the script')
    parser.add_argument('-u', '--user', dest='user', help='Database user', default='postgres')
    parser.add_argument('-p', '--password', dest='password', help='Database password', default='postgres')
    parser.add_argument('-d', '--database', dest='database', help='Database name', default='postgres')
    parser.add_argument('-s', '--host', dest='host', help='Database host', default='localhost')
    parser.add_argument('-t', '--port', dest='port', help='Database port', default='5432')
    parser.add_argument('-o', '--output', dest='output', help='Output file', default='road_length.nc')
    parser.add_argument('-g', '--grid', dest='grid', help='Grid size in degrees', default=0.1)
    parser.add_argument('-c', '--crs', dest='crs', help='Projection of the grid', default='EPSG:3857')
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

    # Create connection to database
    engine = createEngine(args.user, args.password, args.host, args.port, args.database)

    # Read data from database
    data = readData(engine)

    # Create grid
    grid = drawGrid(data, float(args.grid))

    # Calculate lengths
    grid = calculateLengths(data, grid, proj=args.crs)

    # Create xarray dataset
    ds = tableToXarray(grid, args.output)

    # Save map
    if args.map:
        plotLengths(ds)




