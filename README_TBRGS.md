# TBRGS integration starter

This folder is an initial integration scaffold for the COS30019 Assignment 2B
Traffic-Based Route Guidance System. It is designed to sit alongside the
existing Part A source files without modifying the original six algorithms.

## Contents

- `tbrgs/road_graph.py`: SCATS-site and directed road-link data structures.
- `tbrgs/travel_time.py`: converts predicted traffic flow into seconds per road link.
- `tbrgs/top_k_routes.py`: Dijkstra plus Yen's K-shortest loopless paths.
- `tbrgs/config.py`: one place to change lecturer-dependent assumptions.
- `tbrgs/integration_demo.py`: end-to-end demonstration using synthetic data only.
- `tbrgs/tests/test_tbrgs.py`: initial automated tests.
- `data/tbrgs/`: CSV templates for the verified SCATS sites and directed road links.

## Requirements

Python 3.10 or newer. The routing scaffold uses only the Python standard library.

## Run the demonstration

From the repository root:

```bash
python -m tbrgs.integration_demo
```

The demo uses invented intermediate nodes, distances, and mock predictions.
It is only a software-pipeline check. Do not present its route times as actual
Boroondara travel-time results.

## Run the tests

From the repository root:

```bash
python -m unittest discover -s tbrgs/tests -v
```

## CSV input schemas

`data/tbrgs/scats_sites_template.csv`:

```text
scats_id,site_name,latitude,longitude
```

`data/tbrgs/road_links_template.csv`:

```text
from_scats,to_scats,distance_km,controlled_intersections
```

Each road-link row represents one direction. Add a second row for the reverse
direction where it exists. Enter road distance, not straight-line distance.
The CSV templates are intentionally empty until the team's Boroondara road
network data has been verified.

## ML-to-routing interface

The ML team can provide predictions as a dictionary keyed by SCATS ID:

```python
predictions_by_scats = {
    2000: TrafficPrediction(
        scats_id=2000,
        predicted_flow=80,
        unit="vehicles_per_15_minutes",
        model_name="LSTM",
        prediction_time="2006-10-01T08:15:00",
    ),
}
```

The current interface accepts `vehicles_per_15_minutes` or
`vehicles_per_hour`. Confirm that model outputs use the same interval meaning
as the assignment dataset before integrating them.

## Lecturer-dependent settings

The following defaults are provisional and are kept configurable:

- `flow_source="destination"`: follows the assignment brief's reference to
  destination site B, while the conversion guide instead says starting site A.
- `speed_branch="free_flow"`: the quadratic yields two possible speeds for
  some flow values. Confirm the required branch/branches.
- `oversaturation_policy="raise"`: predictions above 1500 vehicles/hour stop
  with an explanatory error rather than silently clamping the flow.

Edit `TBRGSConfig` in `tbrgs/config.py` after receiving lecturer guidance.

## Integration status

This is a routing scaffold, not the completed TBRGS. The team still needs to:

1. populate the CSV files with verified Boroondara SCATS sites and directed road links;
2. connect a real ML predictor for each required SCATS site and departure time;
3. agree how controlled-intersection delays are counted per link;
4. add GUI/input controls and visualisation;
5. expand the tests into at least 15 documented assignment-level test cases,
   including real Boroondara origin/destination pairs.
