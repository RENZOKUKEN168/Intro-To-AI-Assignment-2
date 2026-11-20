# TBRGS: traffic prediction connected to Assignment 2A route search

The command-line integration loads a directed SCATS road network, predicts traffic flow for each network node, converts each directed road link into a travel-time cost, and sends those costs to the Assignment 2A `RouteProblem` and A* search. It also calculates up to five distinct, loopless alternative routes with Yen's algorithm. The A* result and Yen's first route are checked to ensure their shortest travel-time costs agree.

## Run

First create the real network files described below, and train the selected model. Example using Random Forest:

```powershell
python -m tbrgs.app 2000 3002 "2006-10-29 04:00" --model RandomForest --lookback 12 --k-routes 5 --skip-unavailable-sites
```

Other supported model names are `LSTM`, `GRU`, and `BiLSTM`. Neural-model inference requires a compatible TensorFlow installation; Random Forest inference does not.

## Required road-network files

The app requires:

- `data/tbrgs/scats_sites.csv`
- `data/tbrgs/road_links.csv`

The repository includes `data/tbrgs/scats_sites.csv`, populated with the 40 SCATS IDs and location descriptions from the supplied VicRoads SCATS Site Listing. Latitude and longitude are blank because that listing provides map references rather than coordinates. If any node lacks coordinates, the app safely uses a zero heuristic for all nodes. The dataset has three SCATS sites with incomplete 12-step history at the example held-out time `2006-10-29 04:00`; the explicit `--skip-unavailable-sites` option filters those sites and their incident links, while still requiring valid history for the requested origin and destination. The repository still includes only a template for `road_links.csv`; populate it using verified mapping information:

`scats_sites.csv` columns:

```csv
scats_id,site_name,latitude,longitude
```

`road_links.csv` columns:

```csv
from_scats,to_scats,distance_km,controlled_intersections
```

Each row in `road_links.csv` represents one travel direction. Add a reverse edge only where the road is actually traversable in the reverse direction. Do not infer road connectivity or road distance from straight-line coordinates. The supplied traffic-count location catalogue has no explicit SCATS-ID field and must not be treated as a verified SCATS road graph.

## Travel-time conversion

The supplied `docs/Traffic_Flow_to_Travel_Time_Conversion_v1.0.pdf` gives the relationship:

```text
flow = -1.4648375 * speed^2 + 93.75 * speed
```

The model predicts vehicles per 15 minutes, so the app multiplies by four to obtain vehicles per hour before applying the curve. Link travel time is `distance / speed`, converted to seconds, plus 30 seconds for each configured controlled intersection. The speed limit is 60 km/h and the illustrative capacity is 1,500 vehicles/hour at 32 km/h.

The guide says to use the flow at the starting SCATS site, while an assignment brief note in the project indicates destination-site flow. This remains a lecturer-confirmation item; select it explicitly with `--flow-source origin` or `--flow-source destination`. The default remains `destination` to preserve the current brief-based setting.

The two speed branches and treatment of flows above the illustrative capacity are also assumptions. The default oversaturation policy is `raise`, which stops rather than silently inventing a speed. `--oversaturation-policy clamp` is available only if the lecturer approves and the report documents that assumption.

## Current data limitations

The October 2006 workbook and processed prediction dataset are included. The location catalogue is preserved under `data/raw/`, but it does not provide a reliable SCATS-site-to-road-link mapping. The real directed road links, measured distances, and controlled-intersection counts still need to be supplied/verified. Therefore, a genuine end-to-end route cannot be demonstrated until `road_links.csv` is populated with verified directed road connections and distances.

## Tests

```powershell
python -m unittest discover -s tbrgs/tests -v
```

The tests cover flow conversion, directed graph validation, A* integration, and up to five ranked Yen routes. They use small test graphs, not the real Boroondara road network.
