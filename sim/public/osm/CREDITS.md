# Map data

`moss.json` and `route.json` are derived from OpenStreetMap data, © OpenStreetMap contributors, available under the Open Database License (ODbL) 1.0: https://www.openstreetmap.org/copyright

`scripts/osm_corner.py` fetches the data through the Overpass API.

Where the City of San Francisco publishes better numbers, `scripts/sf_open_data.py` takes them from DataSF (https://datasf.org/opendata/, Open Data Commons Public Domain Dedication and License, PDDL 1.0), fetched through the Socrata API at data.sf.gov:

- Building Footprints (`ynuv-fyni`): building heights from the 2010 LiDAR survey (`hgt_median_m`, `hgt_maxcm`), replacing OSM's levels-based estimates.
- Curbs and Islands (`emxt-b6yg`): the real curb lines, for roadway widths and curb positions.
- Sidewalk Widths, 2014 (`4g86-grxu`): sidewalk width per street.
- San Francisco Street Tree Inventory (`tkzw-k3nq`): street tree positions, species and trunk diameter.
- Parking Meters (`8vzz-qzz9`): meter post positions.

Tree heights and crown sizes, ground-floor footprints under bay windows, lane positions and the demo route are our own estimates built on that data. `sources` in `moss.json` lists what was used.
