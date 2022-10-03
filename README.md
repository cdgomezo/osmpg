# osmpg
Singularity container and recipe to extract data from OpenStreetMaps using osmium, osm2pgsql, PostgreSQL, PostGIS, and Python packages: geopandas, sqlalchemy, etc.

## How to use
This version needs sudo priviledges. Working on removing them.
1. Install singularity: https://docs.sylabs.io/guides/3.0/user-guide/installation.html
2. Download the container osmpg.sif
3. Open a shell terminal of the container: `sudo singularity shell --writable osmpg.sif`
4. Execute the scripts

If you want to make the whole installation yourself, follow the recipe osmpg_nosin.def. This works as well as a recipe for building a custom container not dependent on the docker image of PostgreSQL/PostGIS. To build this container you can do: `sudo singularity build osmpg_nosin.sif osmpg_nosin.def` os `sudo singularity build --sandbox osmpg_nosin.sif osmpg_nosin.def` if you want to have access to the installation folders and add commands, packages, etc.
