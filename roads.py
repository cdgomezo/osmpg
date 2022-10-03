from sqlalchemy import create_engine
from pandas import read_sql, DataFrame, merge
from geopandas import read_postgis, GeoDataFrame
from numpy import arange, array
from shapely import geometry

# Function to create grid

def drawGrid(geom_bounds, square_size, proj=None):
    '''
    Creates a grid with squared grid cells around a defined geometry in geografical coordinates or projected if specified.
    square_size: size of the cell in degrees
    '''
    total_bounds = geom_bounds.total_bounds
    minX, minY, maxX, maxY = total_bounds
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

    return grid

# Connect to the database
engine = create_engine('postgresql://postgres:postgres@localhost:5433/osmpg') # Change this to your database connection

# Get the data
gdf = read_postgis("SELECT * FROM highways WHERE type = 'motorway' OR type = 'trunk' OR type = 'primary' OR type = 'secondary' OR type = 'tertiary' OR type = 'motorway_link' OR type = 'trunk_link' OR type = 'primary_link' OR type = 'secondary_link' OR type = 'tertiary_link'", engine, geom_col='geom')

# Create the grid
grid = drawGrid(gdf, 0.1, 'EPSG:3857') # Change the square size and projection, web mercator is used here
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

