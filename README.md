# osmpg
Singularity container and recipe to extract data from OpenStreetMaps using osmium, osm2pgsql, PostgreSQL, PostGIS, and Python packages: geopandas, sqlalchemy, etc.

## How to use
This version needs sudo priviledges. Working on removing them.
1. Install singularity: https://docs.sylabs.io/guides/3.0/user-guide/installation.html
2. Build the container with: `sudo singularity build --sandbox osmpg osmpg_nosin.def`
3. Open a shell terminal of the container: `sudo singularity shell --writable osmpg`
4. Execute the scripts

Alternatively, you can install yourself (without singularity) all the required packages following the recipe in osmpg_nosin.def.
