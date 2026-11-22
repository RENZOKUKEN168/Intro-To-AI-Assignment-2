"""Run an A2A search (A* by default) and up to K Yen routes using traffic-predicted edge costs.

Example:
    python -m tbrgs.app 2000 3002 "2006-10-29 04:00" --model RandomForest --lookback 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ml.predict_model import MODEL_NAMES, WINDOW_SIZES, predict_traffic
from route_problem import RouteProblem
from search import METHODS as SEARCH_METHODS, run_search
from tbrgs.config import TBRGSConfig
from tbrgs.road_graph import build_road_network, load_road_network
from tbrgs.top_k_routes import k_shortest_loopless_paths
from tbrgs.travel_time import build_time_weighted_graph

BASE_DIR = Path(__file__).resolve().parents[1]


def run_tbrgs(
    origin: int,
    destination: int,
    prediction_time: str,
    model_name: str,
    lookback: int,
    sites_csv: Path,
    links_csv: Path,
    flow_source: str = "destination",
    speed_branch: str = "free_flow",
    k_routes: int = 5,
    oversaturation_policy: str = "raise",
    skip_unavailable_sites: bool = False,
    search_method: str = "AS",
):
    """Build predicted edge costs, run an A2A search, and calculate up to K alternatives.

    ``search_method`` selects any Assignment 2A algorithm (``DFS``, ``BFS``,
    ``GBFS``, ``AS``, ``CUS1``, ``CUS2``); A* is the default because it is the
    only one that is optimal on travel-time costs.

    Returns ``(search_result, yen_routes, link_estimates, config, skipped_sites)``.
    Yen routes are ordered by travel time and are distinct and loopless. A*
    remains the Assignment 2A search integration and its cost is checked
    against Yen's best. When explicitly enabled, sites without consecutive
    history are removed together with their incident links.
    """
    search_method = search_method.upper()
    if search_method not in SEARCH_METHODS:
        raise ValueError(
            f"Unknown search method {search_method!r}; choose one of {', '.join(SEARCH_METHODS)}."
        )

    network = load_road_network(sites_csv, links_csv)
    if origin not in network.sites or destination not in network.sites:
        raise ValueError("Origin and destination must be listed in the SCATS sites CSV.")

    predictions = {}
    skipped_sites: list[int] = []
    for site_id in network.node_ids:
        try:
            predictions[site_id] = predict_traffic(
                site_id, prediction_time, model_name=model_name, lookback=lookback
            )
        except ValueError as error:
            history_gap = (
                str(error).startswith("Not enough history for SCATS site")
                or "Historical data must be consecutive 15-minute intervals" in str(error)
            )
            if not skip_unavailable_sites or not history_gap:
                raise
            skipped_sites.append(site_id)

    if origin not in predictions or destination not in predictions:
        raise ValueError(
            "Origin and destination must both have valid prediction history at "
            f"{prediction_time}; unavailable sites: {skipped_sites}."
        )

    if skipped_sites:
        available_ids = set(predictions)
        network = build_road_network(
            [site for site_id, site in network.sites.items() if site_id in available_ids],
            [
                link for link in network.links
                if link.from_scats in available_ids and link.to_scats in available_ids
            ],
        )

    config = TBRGSConfig(
        flow_source=flow_source,
        speed_branch=speed_branch,
        oversaturation_policy=oversaturation_policy,
        k_routes=k_routes,
    )
    weighted_graph, estimates = build_time_weighted_graph(network, predictions, config)

    # If even one site has no coordinate, use a zero heuristic for every node.
    # This keeps A* admissible instead of mixing real and placeholder coordinates.
    coordinates_available = all(
        site.latitude is not None and site.longitude is not None
        for site in network.sites.values()
    )
    if coordinates_available:
        coordinates = {
            site_id: (site.longitude, site.latitude)
            for site_id, site in network.sites.items()
        }
    else:
        coordinates = {site_id: (0.0, 0.0) for site_id in network.node_ids}

    problem = RouteProblem(weighted_graph, origin, [destination], coordinates)
    search_result = run_search(problem, search_method)
    alternatives = k_shortest_loopless_paths(
        weighted_graph, origin, destination, k=config.k_routes
    )

    # Every complete search must agree with Yen on whether a route exists.
    if search_result.goal is None and alternatives:
        raise RuntimeError(f"{search_method} and Yen disagree: Yen found a route but {search_method} did not.")
    if search_result.goal is not None and not alternatives:
        raise RuntimeError(f"{search_method} and Yen disagree: {search_method} found a route but Yen did not.")
    # Only A* is cost-optimal, so only A* must match Yen's best travel time.
    # DFS, BFS, GBFS and the move-count custom searches may return longer routes.
    if (
        search_method == "AS"
        and alternatives
        and abs(search_result.path_cost - alternatives[0].travel_time_seconds) > 1e-6
    ):
        raise RuntimeError("A* and Yen disagree on the shortest travel-time cost.")

    return search_result, alternatives, estimates, config, skipped_sites


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
    parser.add_argument(
        "--method", type=str.upper, choices=tuple(SEARCH_METHODS), default="AS",
        help="Assignment 2A search algorithm for the primary route (default: AS = A*)",
    )
    parser.add_argument("--k-routes", type=int, default=5, help="Maximum number of loopless routes (default: 5)")
    parser.add_argument(
        "--oversaturation-policy", choices=("raise", "clamp"), default="raise",
        help="Use clamp only if approved/documented by the lecturer; default is to stop above capacity.",
    )
    parser.add_argument(
        "--skip-unavailable-sites", action="store_true",
        help="Explicitly exclude sites without consecutive history and their incident links.",
    )
    args = parser.parse_args(argv)
    if args.k_routes < 1:
        parser.error("--k-routes must be at least 1.")

    for path, template in (
        (args.sites, BASE_DIR / "data/tbrgs/scats_sites_template.csv"),
        (args.links, BASE_DIR / "data/tbrgs/road_links_template.csv"),
    ):
        if not path.exists():
            parser.error(f"Required road-network file missing: {path}. Copy/fill {template} first.")

    try:
        result, routes, estimates, config, skipped_sites = run_tbrgs(
            args.origin,
            args.destination,
            args.prediction_time,
            args.model,
            args.lookback,
            args.sites,
            args.links,
            args.flow_source,
            args.speed_branch,
            args.k_routes,
            args.oversaturation_policy,
            args.skip_unavailable_sites,
            args.method,
        )
    except (FileNotFoundError, KeyError, ValueError, ImportError, RuntimeError) as error:
        parser.error(str(error))

    if result.goal is None or not routes:
        print("No route found.")
        return 1

    print(f"Model: {args.model} | Lookback: {args.lookback} x 15 min")
    print(f"Prediction time: {args.prediction_time}")
    print(
        f"Flow source: {config.flow_source} | Speed branch: {config.speed_branch} | "
        f"Oversaturation: {config.oversaturation_policy}"
    )
    print(f"A2A {args.method} route: {' -> '.join(map(str, result.path))}")
    print(
        f"A2A {args.method} travel time: {result.path_cost:.1f} seconds "
        f"({result.path_cost / 60:.2f} minutes)"
    )
    print(f"Search nodes created: {result.nodes_created}")
    print(f"Predicted links: {len(estimates)}")
    if skipped_sites:
        print(f"Skipped sites without consecutive history: {', '.join(map(str, skipped_sites))}")
    print(f"\nTop {len(routes)} loopless routes (Yen, ranked by travel time):")
    for index, route in enumerate(routes, start=1):
        path = " -> ".join(map(str, route.nodes))
        print(
            f"{index}. {path} | {route.travel_time_seconds:.1f} seconds "
            f"({route.travel_time_minutes:.2f} minutes)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
