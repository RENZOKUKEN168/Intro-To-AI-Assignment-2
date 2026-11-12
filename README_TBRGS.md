# TBRGS: traffic prediction connected to A2A route search

The project contains an end-to-end command-line adapter at `tbrgs/app.py`. It loads a directed SCATS road network, predicts the requested 15-minute traffic flow for the selected model and lookback, converts each directed link prediction into travel time, and passes those seconds to the Assignment 2A `RouteProblem` and A* search.

## Run

After preparing data and training the models (see `README.md`):

```sh
python -m tbrgs.app 2000 3002 "2006-10-27 08:00" --model GRU --lookback 8
```

The app expects populated files at `data/tbrgs/scats_sites.csv` and `data/tbrgs/road_links.csv`. Copy the schemas from `scats_sites_template.csv` and `road_links_template.csv`. Add only verified SCATS site IDs and directed road links, with measured road distance and controlled-intersection count. Add a reverse edge only when the real road is usable in that direction. The repo currently has no verified road-link mapping, so the synthetic demo is not a substitute for filling these files.

## Prediction and cost conversion

Each model predicts vehicles in the next 15-minute interval for one SCATS site. The configurable `TBRGSConfig` selects whether link flow comes from the origin or destination site and which speed branch to use. Current defaults are `destination` and `free_flow`; validate both against the course's conversion instructions before reporting final route results. Flow is converted from vehicles per 15 minutes to vehicles per hour before applying the supplied speed curve. Route weights are the resulting travel time in seconds, which A2A A* minimizes.

## Tests

```sh
python -m unittest discover -s tbrgs/tests -v
```

The tests cover data-to-cost conversion, directed road links, route ranking, and a path through A2A's A* using dynamic travel-time weights.
