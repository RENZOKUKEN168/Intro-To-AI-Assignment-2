# Raw source data

- `Scats Data October 2006.xls`: original SCATS traffic observations used by the existing preprocessing pipeline.
- `SCATSSiteListingSpreadsheet_VicRoads.xls`: SCATS site-number listing. The 40 IDs present in the October 2006 processed dataset were matched to their location descriptions and placed in `data/tbrgs/scats_sites.csv`.
- `Traffic_Count_Locations_with_LONG_LAT.csv`: supplementary Victorian traffic-count location catalogue. Its columns do not include an explicit SCATS site-number field. It is preserved for reference but is not used to assign SCATS coordinates or infer road connections.

Keep these original source files unchanged. Processed traffic data and sequence files are stored separately under `data/processed/`.
