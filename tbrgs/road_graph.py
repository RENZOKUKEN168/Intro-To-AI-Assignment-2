"""SCATS-site road-network data structures and CSV loading helpers.

Each RoadLink row represents one travel direction. If a road is usable in
both directions, include a second row with the endpoints reversed. This avoids
assuming that the two directions have identical distance or signal delay.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ScatsSite:
    scats_id: int
    site_name: str = ""
    latitude: float | None = None
    longitude: float | None = None

    def __post_init__(self) -> None:
        if type(self.scats_id) is not int:
            raise ValueError("SCATS IDs must be integers.")
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("Provide both latitude and longitude, or neither.")
        if self.latitude is not None:
            if not math.isfinite(self.latitude) or not -90 <= self.latitude <= 90:
                raise ValueError("Latitude must be finite and between -90 and 90.")
            if not math.isfinite(self.longitude) or not -180 <= self.longitude <= 180:
                raise ValueError("Longitude must be finite and between -180 and 180.")


@dataclass(frozen=True)
class RoadLink:
    from_scats: int
    to_scats: int
    distance_km: float
    controlled_intersections: int = 0

    def __post_init__(self) -> None:
        if type(self.from_scats) is not int or type(self.to_scats) is not int:
            raise ValueError("Road-link SCATS IDs must be integers.")
        if self.from_scats == self.to_scats:
            raise ValueError("A road link must connect two different SCATS sites.")
        if not math.isfinite(self.distance_km) or self.distance_km <= 0:
            raise ValueError("Road-link distance must be a finite positive number.")
        if (
            type(self.controlled_intersections) is not int
            or self.controlled_intersections < 0
        ):
            raise ValueError("controlled_intersections must be a nonnegative integer.")


@dataclass(frozen=True)
class RoadNetwork:
    sites: dict[int, ScatsSite]
    links: tuple[RoadLink, ...]

    @property
    def node_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.sites))

    def outgoing_links(self, scats_id: int) -> tuple[RoadLink, ...]:
        if scats_id not in self.sites:
            raise KeyError(f"Unknown SCATS site: {scats_id}")
        return tuple(link for link in self.links if link.from_scats == scats_id)


def build_road_network(
    sites: Iterable[ScatsSite], links: Iterable[RoadLink]
) -> RoadNetwork:
    """Validate and assemble a directed road network."""

    sites_by_id: dict[int, ScatsSite] = {}
    for site in sites:
        if site.scats_id in sites_by_id:
            raise ValueError(f"Duplicate SCATS site ID: {site.scats_id}")
        sites_by_id[site.scats_id] = site

    if not sites_by_id:
        raise ValueError("The road network must contain at least one SCATS site.")

    links_tuple = tuple(links)
    seen_edges: set[tuple[int, int]] = set()
    for link in links_tuple:
        if link.from_scats not in sites_by_id or link.to_scats not in sites_by_id:
            raise ValueError(
                f"Road link {link.from_scats}->{link.to_scats} references "
                "a SCATS site that is not in the site list."
            )
        edge = (link.from_scats, link.to_scats)
        if edge in seen_edges:
            raise ValueError(f"Duplicate directed road link: {edge[0]}->{edge[1]}")
        seen_edges.add(edge)

    return RoadNetwork(sites=sites_by_id, links=links_tuple)


def load_road_network(
    sites_csv: str | Path, links_csv: str | Path
) -> RoadNetwork:
    """Load the documented CSV schemas used by the integration package.

    sites CSV columns:
        scats_id,site_name,latitude,longitude

    links CSV columns:
        from_scats,to_scats,distance_km,controlled_intersections

    Latitude/longitude are optional metadata. Road distance must be supplied
    separately; this loader never substitutes straight-line distance for road
    distance.
    """

    site_rows: list[ScatsSite] = []
    with Path(sites_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"scats_id", "site_name", "latitude", "longitude"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Sites CSV must contain: scats_id, site_name, latitude, longitude."
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                lat_text = (row.get("latitude") or "").strip()
                lon_text = (row.get("longitude") or "").strip()
                latitude = float(lat_text) if lat_text else None
                longitude = float(lon_text) if lon_text else None
                site_rows.append(
                    ScatsSite(
                        scats_id=int(row["scats_id"]),
                        site_name=(row.get("site_name") or "").strip(),
                        latitude=latitude,
                        longitude=longitude,
                    )
                )
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid sites CSV row {line_number}: {error}") from error

    link_rows: list[RoadLink] = []
    with Path(links_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "from_scats",
            "to_scats",
            "distance_km",
            "controlled_intersections",
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Links CSV must contain: from_scats, to_scats, distance_km, "
                "controlled_intersections."
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                link_rows.append(
                    RoadLink(
                        from_scats=int(row["from_scats"]),
                        to_scats=int(row["to_scats"]),
                        distance_km=float(row["distance_km"]),
                        controlled_intersections=int(
                            row["controlled_intersections"] or 0
                        ),
                    )
                )
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid links CSV row {line_number}: {error}") from error

    return build_road_network(site_rows, link_rows)
