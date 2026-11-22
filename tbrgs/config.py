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


# ---------------------------------------------------------------------------
# GUI / command-line defaults loaded from the JSON configuration file.
# ---------------------------------------------------------------------------

import json
from pathlib import Path
from typing import Any

DEFAULTS_FILE = Path(__file__).resolve().parents[1] / "config.json"

# Built-in fallbacks, used for any key missing from config.json.
BUILTIN_DEFAULTS: dict[str, Any] = {
    "origin": 2000,
    "destination": 3002,
    "prediction_time": "2006-10-27 08:00",
    "model": "LSTM",
    "lookback": 8,
    "search_method": "AS",
    "k_routes": 5,
    "flow_source": "destination",
    "speed_branch": "free_flow",
    "oversaturation_policy": "raise",
    "skip_unavailable_sites": False,
    "sites_csv": "data/tbrgs/scats_sites.csv",
    "links_csv": "data/tbrgs/road_links.csv",
}


def load_defaults(path: Path | str = DEFAULTS_FILE) -> dict[str, Any]:
    """Return the GUI defaults: built-in values overridden by the JSON file.

    A missing file falls back to the built-in values. Unknown keys and invalid
    JSON raise ``ValueError`` so a typo in the file is reported, not ignored.
    Relative CSV paths are resolved against the repository root.
    """
    defaults = dict(BUILTIN_DEFAULTS)
    path = Path(path)
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON in {path}: {error}") from error
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} must contain a JSON object.")
        unknown = set(loaded) - set(BUILTIN_DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown keys in {path}: {', '.join(sorted(unknown))}")
        defaults.update(loaded)

    root = DEFAULTS_FILE.parent
    for key in ("sites_csv", "links_csv"):
        csv_path = Path(defaults[key])
        defaults[key] = csv_path if csv_path.is_absolute() else root / csv_path
    return defaults
