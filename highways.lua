local tables = {}

tables.highways = osm2pgsql.define_way_table('highways', {
    { column = 'type', type = 'text' },
    { column = 'lanes', type = 'int8'},
    { column = 'geom', type = 'linestring', projection = 4326 },
})

function osm2pgsql.process_way(object)
    if object.tags.highway then
        tables.highways:add_row{ type = object.tags.highway,
                                 lanes = object.tags.lanes,
                                 geom = { create = 'line' } }
    end
end
