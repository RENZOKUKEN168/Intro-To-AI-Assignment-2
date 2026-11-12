"""Run an end-to-end TBRGS routing demonstration using synthetic data only.

The node IDs 2000 and 3002 mirror the example endpoints in the assignment
brief. The intermediate nodes, road distances, and traffic predictions below
are invented demonstration values, not Boroondara road-network data.
"""

from .config import TBRGSConfig
from .road_graph import RoadLink, ScatsSite, build_road_network
from .travel_time import TrafficPrediction, build_time_weighted_graph
from .top_k_routes import k_shortest_loopless_paths


def main() -> None:
    # Synthetic network used only to verify the software pipeline.
    sites = [
        ScatsSite(2000, "Example origin"),
        ScatsSite(9901, "Example junction A"),
        ScatsSite(9902, "Example junction B"),
        ScatsSite(9903, "Example junction C"),
        ScatsSite(3002, "Example destination"),
    ]
    links = [
        RoadLink(2000, 9901, 1.2, controlled_intersections=1),
        RoadLink(9901, 3002, 1.1, controlled_intersections=1),
        RoadLink(2000, 9902, 1.0, controlled_intersections=0),
        RoadLink(9902, 3002, 1.8, controlled_intersections=1),
        RoadLink(2000, 9903, 1.4, controlled_intersections=1),
        RoadLink(9903, 3002, 1.0, controlled_intersections=0),
        RoadLink(9901, 9902, 0.6, controlled_intersections=0),
    ]
    network = build_road_network(sites, links)

    # Synthetic model output: 15-minute vehicle counts, not real predictions.
    mock_predictions = {
        2000: TrafficPrediction(2000, 70, model_name="MOCK"),
        9901: TrafficPrediction(9901, 95, model_name="MOCK"),
        9902: TrafficPrediction(9902, 115, model_name="MOCK"),
        9903: TrafficPrediction(9903, 80, model_name="MOCK"),
        3002: TrafficPrediction(3002, 100, model_name="MOCK"),
    }

    config = TBRGSConfig(
        flow_source="destination",
        speed_branch="free_flow",
        oversaturation_policy="raise",
        k_routes=5,
    )
    weighted_graph, edge_estimates = build_time_weighted_graph(
        network, mock_predictions, config
    )
    routes = k_shortest_loopless_paths(
        weighted_graph, source=2000, destination=3002, k=config.k_routes
    )

    print("TBRGS INTEGRATION DEMO (SYNTHETIC DATA ONLY)")
    print("Origin: 2000 | Destination: 3002")
    print("Traffic source policy:", config.flow_source)
    print("Speed branch:", config.speed_branch)
    print("\nEstimated directed-link costs:")
    for (start, end), estimate in sorted(edge_estimates.items()):
        print(
            f"  {start} -> {end}: {estimate.total_seconds:.1f} sec "
            f"({estimate.speed_kmh:.1f} km/h, "
            f"{estimate.flow_vehicles_per_hour:.0f} veh/h)"
        )

    print("\nTop routes:")
    if not routes:
        print("  No route found.")
    for rank, route in enumerate(routes, start=1):
        route_text = " -> ".join(str(node) for node in route.nodes)
        print(
            f"  {rank}. {route_text} | "
            f"{route.travel_time_seconds:.1f} sec | "
            f"{route.travel_time_minutes:.2f} min"
        )

    print("\nNOTE: Replace every synthetic road link and mock prediction with the")
    print("verified Boroondara network and ML model outputs before reporting results.")


if __name__ == "__main__":
    main()
