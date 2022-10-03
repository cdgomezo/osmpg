# osmpg
Singularity container and recipe to extract data from OpenStreetMaps using osmium, osm2pgsql, PostgreSQL, PostGIS, and Python packages: geopandas, sqlalchemy, etc.

## How to use
This version needs sudo priviledges. Working on removing them.
1. Install singularity: https://docs.sylabs.io/guides/3.0/user-guide/installation.html
2. Build the container with: `sudo singularity build --sandbox osmpg osmpg.def`. Alternatively, you can install yourself (without singularity) all the required packages following the recipe in osmpg.def.
3. Open a shell terminal of the container: `sudo singularity shell --writable osmpg`
4. Initialize the database: (When creating password choose 'postgres' for simplicity)
```
su - postgres
service postgresql start
psql
\password postgres
CREATE DATABASE osmpg;
\c osmpg
CREATE EXTENSION postgis;
CREATE EXTENSION hstore;
\q
```
5. Check the port where psql is running: `service postgresql status` (Usually 5432 but could be different)
6. `osmium tags-filter -v -o germany-hwy.osm.pbf germany-latest.osm.pbf w/highway `
7. `osm2pgsql -d osmpg -P 5433 -O flex -S highways.lua germany-hwy.osm.pbf` (If psql is not running on port 5432 replace here)
8. Run python script.

## Documentation

osmium: https://osmcode.org/libosmium/
osm2pgsql: https://osm2pgsql.org/
