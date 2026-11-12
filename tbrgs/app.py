"""Run the A2A A* algorithm with traffic-predicted travel-time edge costs."""

from __future__ import annotations

import argparse
from pathlib import Path

from inform_search import astar_search
from ml.predict_model import MODEL_NAMES, WINDOW_SIZES, predict_traffic
from route_problem import RouteProblem
from tbrgs.config import TBRGSConfig
from tbrgs.road_graph import load_road_network
from tbrgs.travel_time import build_time_weighted_graph

BASE_DIR = Path(__file__).resolve().parents[1]


def run_tbrgs(origin: int, destination: int, prediction_time: str, model_name: str,
              lookback: int, sites_csv: Path, links_csv: Path,
              flow_source: str, speed_branch: str):
    network = load_road_network(sites_csv, links_csv)
    if origin not in network.sites or destination not in network.sites:
        raise ValueError("Origin and destination must be listed in the SCATS sites CSV.")

    predictions = {
        site_id: predict_traffic(
            site_id, prediction_time, model_name=model_name, lookback=lookback
        )
        for site_id in network.node_ids
    }
    config = TBRGSConfig(flow_source=flow_source, speed_branch=speed_branch)
    weighted_graph, estimates = build_time_weighted_graph(network, predictions, config)
    coordinates = {}
    for site_id, site in network.sites.items():
        if site.latitude is None or site.longitude is None:
            coordinates[site_id] = (0.0, 0.0)
        else:
            coordinates[site_id] = (site.longitude, site.latitude)

    # The traffic-derived seconds become RouteProblem edge costs; this invokes
    # the Assignment 2A A* implementation on the integrated graph.
    problem = RouteProblem(weighted_graph, origin, [destination], coordinates)
    result = astar_search(problem)
    return result, estimates, config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("origin", type=int)
    parser.add_argument("destination", type=int)
    parser.add_argument("prediction_time", help="Local timestamp on a 15-minute boundary")
    parser.add_argument("--model", choices=MODEL_NAMES, default="GRU")
    parser.add_argument("--lookback", type=int, choices=WINDOW_SIZES, default=8)
    parser.add_argument("--sites", type=Path, default=BASE_DIR / "data/tbrgs/scats_sites.csv")
    parser.add_argument("--links", type=Path, default=BASE_DIR / "data/tbrgs/road_links.csv")
    parser.add_argument("--flow-source", choices=("origin", "destination"), default="destination")
    parser.add_argument("--speed-branch", choices=("free_flow", "congested"), default="free_flow")
    args = parser.parse_args(argv)
    for path, template in (
        (args.sites, BASE_DIR / "data/tbrgs/scats_sites_template.csv"),
        (args.links, BASE_DIR / "data/tbrgs/road_links_template.csv"),
    ):
        if not path.exists():
            parser.error(f"Required road-network file missing: {path}. Copy/fill {template} first.")
    try:
        result, estimates, config = run_tbrgs(
            args.origin, args.destination, args.prediction_time, args.model,
            args.lookback, args.sites, args.links, args.flow_source, args.speed_branch,
        )
    except (FileNotFoundError, KeyError, ValueError, ImportError) as error:
        parser.error(str(error))
    if result.goal is None:
        print("No route found.")
        return 1
    route = " -> ".join(map(str, result.path))
    print(f"Method: A2A A* | Model: {args.model} | Lookback: {args.lookback} x 15 min")
    print(f"Departure/prediction time: {args.prediction_time}")
    print(f"Flow source: {config.flow_source} | Speed branch: {config.speed_branch}")
    print(f"Route: {route}")
    print(f"Estimated travel time: {result.path_cost:.1f} seconds ({result.path_cost / 60:.2f} minutes)")
    print(f"Search nodes created: {result.nodes_created}")
    print(f"Predicted links: {len(estimates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
