from sqlalchemy import create_engine
from pandas import merge
from geopandas import read_postgis, GeoDataFrame
from numpy import arange, array
from shapely import geometry
from xarray import Dataset

# Function to create grid

def drawGrid(geom_bounds, square_size, proj=None):
    '''
    Creates a grid with squared grid cells around a defined geometry in geografical coordinates or projected if specified.
    square_size: size of the cell in degrees
    '''
    total_bounds = geom_bounds.total_bounds
    minX, minY, maxX, maxY = total_bounds

    minX = int(minX)
    minY = int(minY)
    maxX = int(maxX) + 1
    maxY = int(maxY) + 1
    
    x, y = (minX, minY)
    geom_array = []

    while y <= maxY:
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

    return grid, [minX, minY, maxX, maxY]

# Connect to the database
engine = create_engine('postgresql://postgres:postgres@localhost:5433/osmpg') # Change this to your database connection

# Get the data
gdf = read_postgis("SELECT * FROM highways WHERE type = 'motorway' OR type = 'trunk' OR type = 'primary' OR type = 'secondary' OR type = 'tertiary' OR type = 'motorway_link' OR type = 'trunk_link' OR type = 'primary_link' OR type = 'secondary_link' OR type = 'tertiary_link'", engine, geom_col='geom')

# Create the grid
grid, total_bounds = drawGrid(gdf, 0.1, 'EPSG:3857') # Change the square size and projection, web mercator is used here
grid.loc[:,'cell_idx'] = arange(0,grid.shape[0]) # Preparing grid to spatial join with highways

# Calculate the length of the roads in each grid cell
gdf_proj = gdf.to_crs('EPSG:3857') # Project the roads to the same projection as the grid
gdf_proj.loc[:,'length'] = gdf_proj.length

# Spatial join
gdf_grid = gdf_proj.sjoin(grid, how='left', op='intersects')

# Calculate the length of the roads in each grid cell
gdf_grid = gdf_grid.groupby('cell_idx').sum().reset_index()

# Merge the grid with the length of the roads
grid = merge(grid, gdf_grid, on='cell_idx', how='left')

# Creating netCDF file
minX, minY, maxX, maxY = total_bounds
lon = arange(minX+0.1/2, maxX+0.1, 0.1)
lat = arange(minY+0.1/2, maxY, 0.1)

# Create matrix with the length of the roads in each grid cell
road_length = array(grid['length'].values).reshape(len(lat), len(lon))

# Save the netCDF file
ds = Dataset({'road_length': (['lat', 'lon'], road_length[:,:])}, coords={'lat': lat, 'lon': lon})
ds.attrs['title'] = 'Road length'
ds.attrs['description'] = 'Road length in Germany'
ds.attrs['units'] = 'm'
ds.attrs['projection'] = 'EPSG:4326'
ds.attrs['year'] = 2019
ds.to_netcdf('road_length.nc')


