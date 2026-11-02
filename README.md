# viision-data

Street and boundary data for the viision app, built from OpenStreetMap.

Once a month GitHub Actions downloads Italy from Geofabrik, keeps roads and administrative
boundaries, and publishes ready files in Releases (tag `italy`):

- `it-index.json` — all areas of the country with total street length and simplified outlines
- `region-<id>.vpkg` — one file per region: exact boundaries and streets of its municipalities

Data © OpenStreetMap contributors, available under the Open Database License (ODbL).
