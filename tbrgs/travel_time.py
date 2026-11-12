"""Convert SCATS traffic predictions into estimated directed-link travel time.

The conversion curve is taken from the supplied "Traffic Flow to Travel Time
Conversion" guide:

    flow = -1.4648375 * speed**2 + 93.75 * speed

The flow input to the curve is vehicles/hour. Dataset/model outputs from a
single 15-minute interval therefore need multiplying by four before use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from .config import TBRGSConfig
from .road_graph import RoadNetwork


@dataclass(frozen=True)
class TrafficPrediction:
    scats_id: int
    predicted_flow: float
    unit: str = "vehicles_per_15_minutes"
    model_name: str = "unspecified"
    prediction_time: str | None = None

    def __post_init__(self) -> None:
        if type(self.scats_id) is not int:
            raise ValueError("Prediction SCATS ID must be an integer.")
        if not math.isfinite(self.predicted_flow) or self.predicted_flow < 0:
            raise ValueError("Predicted flow must be finite and nonnegative.")
        if self.unit not in ("vehicles_per_15_minutes", "vehicles_per_hour"):
            raise ValueError(
                "Prediction unit must be 'vehicles_per_15_minutes' "
                "or 'vehicles_per_hour'."
            )


@dataclass(frozen=True)
class LinkTravelTimeEstimate:
    distance_km: float
    flow_vehicles_per_hour: float
    speed_kmh: float
    driving_seconds: float
    signal_delay_seconds: float
    total_seconds: float


def flow_to_vehicles_per_hour(flow: float, unit: str) -> float:
    """Normalise a predicted 15-minute count or hourly flow into vehicles/hour."""

    if not math.isfinite(flow) or flow < 0:
        raise ValueError("Traffic flow must be finite and nonnegative.")
    if unit == "vehicles_per_15_minutes":
        return flow * 4.0
    if unit == "vehicles_per_hour":
        return flow
    raise ValueError(
        "Unsupported flow unit. Use 'vehicles_per_15_minutes' or 'vehicles_per_hour'."
    )


def flow_to_speed_kmh(
    flow_vehicles_per_hour: float,
    *,
    speed_limit_kmh: float = 60.0,
    capacity_flow_vehicles_per_hour: float = 1500.0,
    branch: str = "free_flow",
    oversaturation_policy: str = "raise",
) -> float:
    """Solve the supplied quadratic curve for speed in km/h.

    For flows below the speed-limit threshold, speed is capped at the stated
    speed limit. Above that threshold, the quadratic has two possible speeds;
    ``branch`` selects the high-speed (free-flow) or low-speed (congested) root.

    Flows above the assignment guide's approximate 1500 vehicles/hour capacity
    raise by default. ``clamp`` is available as an explicit, configurable policy
    but should only be used if the lecturer approves it.
    """

    if not math.isfinite(flow_vehicles_per_hour) or flow_vehicles_per_hour < 0:
        raise ValueError("Hourly traffic flow must be finite and nonnegative.")
    if not math.isfinite(speed_limit_kmh) or speed_limit_kmh <= 0:
        raise ValueError("Speed limit must be finite and greater than zero.")
    if (
        not math.isfinite(capacity_flow_vehicles_per_hour)
        or capacity_flow_vehicles_per_hour <= 0
    ):
        raise ValueError("Capacity flow must be finite and greater than zero.")
    if branch not in ("free_flow", "congested"):
        raise ValueError("branch must be 'free_flow' or 'congested'.")
    if oversaturation_policy not in ("raise", "clamp"):
        raise ValueError("oversaturation_policy must be 'raise' or 'clamp'.")

    capacity = capacity_flow_vehicles_per_hour
    flow = flow_vehicles_per_hour
    if flow > capacity:
        if oversaturation_policy == "raise":
            raise ValueError(
                f"Predicted flow {flow:.2f} vehicles/hour exceeds the configured "
                f"capacity of {capacity:.2f}. Lecturer confirmation is needed "
                "before choosing an oversaturation treatment."
            )
        flow = capacity

    # The supplied guide gives q = -a*v^2 + b*v, with a and b below.
    a = 1.4648375
    b = 93.75

    # Flow at the speed-limit line. Below this, the model caps speed at 60 km/h.
    speed_limit_flow = -a * speed_limit_kmh**2 + b * speed_limit_kmh
    if flow <= speed_limit_flow:
        return speed_limit_kmh

    discriminant = b**2 - 4.0 * a * flow
    if discriminant < 0:
        # A small negative value can occur from floating-point rounding exactly
        # at a theoretical curve maximum.
        if discriminant > -1e-8:
            discriminant = 0.0
        else:
            raise ValueError("Traffic flow has no real speed solution on the curve.")

    root = math.sqrt(discriminant)
    numerator = b + root if branch == "free_flow" else b - root
    speed = numerator / (2.0 * a)
    return min(speed_limit_kmh, max(0.0, speed))


def estimate_link_travel_time(
    distance_km: float,
    predicted_flow: float,
    *,
    flow_unit: str = "vehicles_per_15_minutes",
    controlled_intersections: int = 0,
    config: TBRGSConfig | None = None,
) -> LinkTravelTimeEstimate:
    """Estimate link travel time in seconds, including explicit signal delays."""

    config = config or TBRGSConfig()
    if not math.isfinite(distance_km) or distance_km <= 0:
        raise ValueError("Road distance must be finite and greater than zero.")
    if type(controlled_intersections) is not int or controlled_intersections < 0:
        raise ValueError("controlled_intersections must be a nonnegative integer.")

    hourly_flow = flow_to_vehicles_per_hour(predicted_flow, flow_unit)
    speed = flow_to_speed_kmh(
        hourly_flow,
        speed_limit_kmh=config.speed_limit_kmh,
        capacity_flow_vehicles_per_hour=config.capacity_flow_vehicles_per_hour,
        branch=config.speed_branch,
        oversaturation_policy=config.oversaturation_policy,
    )
    driving_seconds = (distance_km / speed) * 3600.0
    signal_delay = controlled_intersections * config.signal_delay_seconds
    total_seconds = driving_seconds + signal_delay

    return LinkTravelTimeEstimate(
        distance_km=distance_km,
        flow_vehicles_per_hour=hourly_flow,
        speed_kmh=speed,
        driving_seconds=driving_seconds,
        signal_delay_seconds=signal_delay,
        total_seconds=total_seconds,
    )


def build_time_weighted_graph(
    network: RoadNetwork,
    predictions_by_scats: Mapping[int, TrafficPrediction],
    config: TBRGSConfig | None = None,
) -> tuple[dict[int, dict[int, float]], dict[tuple[int, int], LinkTravelTimeEstimate]]:
    """Build ``SCATS A -> SCATS B: estimated seconds`` from a traffic snapshot.

    ``predictions_by_scats`` is the shared interface expected from the ML team.
    Each dictionary key must equal the SCATS ID stored in its prediction.
    """

    config = config or TBRGSConfig()
    weighted_graph: dict[int, dict[int, float]] = {
        scats_id: {} for scats_id in network.node_ids
    }
    estimates: dict[tuple[int, int], LinkTravelTimeEstimate] = {}

    for link in network.links:
        prediction_site = (
            link.from_scats if config.flow_source == "origin" else link.to_scats
        )
        prediction = predictions_by_scats.get(prediction_site)
        if prediction is None:
            raise KeyError(
                f"Missing traffic prediction for SCATS site {prediction_site}, "
                f"needed for link {link.from_scats}->{link.to_scats}."
            )
        if prediction.scats_id != prediction_site:
            raise ValueError(
                f"Prediction mapping key {prediction_site} does not match "
                f"prediction SCATS ID {prediction.scats_id}."
            )

        estimate = estimate_link_travel_time(
            link.distance_km,
            prediction.predicted_flow,
            flow_unit=prediction.unit,
            controlled_intersections=link.controlled_intersections,
            config=config,
        )
        weighted_graph[link.from_scats][link.to_scats] = estimate.total_seconds
        estimates[(link.from_scats, link.to_scats)] = estimate

    return weighted_graph, estimates
