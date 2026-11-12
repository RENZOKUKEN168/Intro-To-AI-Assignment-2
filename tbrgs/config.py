"""Central configuration for the TBRGS routing prototype.

Two assumptions are intentionally configurable because the assignment
materials need lecturer clarification:
1. Which SCATS site supplies the flow for a directed A -> B road link.
2. Which branch of the simplified flow-speed curve should be used, and how
   to handle predictions above the stated capacity.

The current defaults are provisional, not confirmed lecturer decisions.
"""

from dataclasses import dataclass
from typing import Literal

FlowSource = Literal["origin", "destination"]
SpeedBranch = Literal["free_flow", "congested"]
OversaturationPolicy = Literal["raise", "clamp"]


@dataclass(frozen=True)
class TBRGSConfig:
    """Runtime assumptions used when converting predictions into route costs."""

    # Assignment brief states that SCATS flow observations are at 15-minute
    # intervals. This is the expected default output unit from the ML model.
    default_flow_unit: str = "vehicles_per_15_minutes"

    # PROVISIONAL: the assignment brief refers to destination site B, while
    # the conversion guide refers to starting site A. Confirm with the lecturer.
    flow_source: FlowSource = "destination"

    # PROVISIONAL: assumes the free-flow/high-speed branch for the prototype.
    # Confirm whether the implementation should use one or both branches.
    speed_branch: SpeedBranch = "free_flow"

    # Do not silently invent a speed when predicted flow exceeds capacity.
    # Change to "clamp" only if the lecturer approves that treatment.
    oversaturation_policy: OversaturationPolicy = "raise"

    speed_limit_kmh: float = 60.0
    capacity_flow_vehicles_per_hour: float = 1500.0
    signal_delay_seconds: float = 30.0
    k_routes: int = 5

    def __post_init__(self) -> None:
        if self.flow_source not in ("origin", "destination"):
            raise ValueError("flow_source must be 'origin' or 'destination'.")
        if self.speed_branch not in ("free_flow", "congested"):
            raise ValueError("speed_branch must be 'free_flow' or 'congested'.")
        if self.oversaturation_policy not in ("raise", "clamp"):
            raise ValueError("oversaturation_policy must be 'raise' or 'clamp'.")
        if self.default_flow_unit not in (
            "vehicles_per_15_minutes",
            "vehicles_per_hour",
        ):
            raise ValueError(
                "default_flow_unit must be 'vehicles_per_15_minutes' "
                "or 'vehicles_per_hour'."
            )
        if self.speed_limit_kmh <= 0:
            raise ValueError("speed_limit_kmh must be greater than zero.")
        if self.capacity_flow_vehicles_per_hour <= 0:
            raise ValueError("Capacity flow must be greater than zero.")
        if self.signal_delay_seconds < 0:
            raise ValueError("signal_delay_seconds cannot be negative.")
        if self.k_routes < 1:
            raise ValueError("k_routes must be at least 1.")
