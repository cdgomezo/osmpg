local tables = {}

tables.highways = osm2pgsql.define_way_table('highways', {
    { column = 'type', type = 'text' },
    { column = 'lanes', type = 'int8'},
    { column = 'geom', type = 'linestring', projection = 4326 },
})

tables.boundaries = osm2pgsql.define_area_table('boundaries', {
    { column = 'boundary', type = 'text' },
    { column = 'admin_level', type = 'int8' },
    { column = 'name', type = 'text' },
    { column = 'geom', type = 'geometry', projection = 4326 },
})

function osm2pgsql.process_way(object)
    if object.tags.highway then
        tables.highways:add_row{ type = object.tags.highway,
                                 lanes = object.tags.lanes,
                                 geom = { create = 'line' } }
    end
end

function osm2pgsql.process_relation(object)
    if (object.tags.boundary == 'administrative') and (object.tags.admin_level == '4') then
        tables.boundaries:add_row{
            boundary = object.tags.boundary,
            admin_level = object.tags.admin_level,
            name = object.tags.name,
            geom = { create = 'area' }
        }
    end
end
