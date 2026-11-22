"""Build the Boroondara SCATS road network used by the route planner.

Run from the repository root:

    python -m tbrgs.build_road_network

Outputs
-------
data/tbrgs/scats_sites.csv
    Every SCATS site with a latitude/longitude. A site's position is the mean
    of its detector approaches in the VicRoads October 2006 workbook
    (``NB_LATITUDE``/``NB_LONGITUDE``); placeholder ``0, 0`` rows are ignored.

data/tbrgs/road_links.csv
    Directed road links between neighbouring SCATS sites.

How the links were chosen
-------------------------
Each SCATS site is a signalised intersection of named roads (the workbook's
``Location`` column names every approach, e.g. ``BURKE_RD N of TOORAK_RD``).
``ROAD_SEGMENTS`` below lists, for each road, the SCATS sites it passes
through in order along the road. Consecutive sites on the same road are linked
in both directions, because every road in the list is two-way.

A few links follow a road that changes name at an intersection (for example
Victoria St becomes Barkers Rd at the Yarra, Warrigal Rd becomes Union Rd
north of Canterbury Rd). These are commented where they appear.

Assumptions (state these in the report)
---------------------------------------
* Distance is the great-circle (haversine) distance between the two sites.
  The linked sites lie on the same, mostly straight, arterial road, so this is
  a close approximation of road distance, but it is not a measured distance.
* ``controlled_intersections`` is 1 for every link: the 30-second delay of the
  signalised SCATS intersection at the end of the link.
* SCATS coordinates in this dataset are known to be slightly offset from real
  map positions (see the assignment brief); the offset is similar for all
  sites, so relative distances are still usable.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict, deque
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_WORKBOOK = BASE_DIR / "data" / "raw" / "Scats Data October 2006.xls"
SITES_CSV = BASE_DIR / "data" / "tbrgs" / "scats_sites.csv"
LINKS_CSV = BASE_DIR / "data" / "tbrgs" / "road_links.csv"

# SCATS sites in order along each road. Consecutive pairs become two-way links.
ROAD_SEGMENTS: dict[str, list[int]] = {
    # North-south arterials (listed north to south).
    "Burke Rd": [2825, 4030, 4032, 4034, 4035, 3120, 4040, 4043],
    "Warrigal Rd / Union Rd": [2200, 3126, 3682, 2000, 3685, 970],  # Union Rd continues as Warrigal Rd at Canterbury Rd
    "Balwyn Rd": [3180, 4057, 4063, 3127],
    "Glenferrie Rd": [4324, 4264, 4270],
    "Tooronga Rd": [4272, 4273],
    "Trafalgar Rd": [3804, 3812],
    "Bulleen Rd": [2827, 4051],
    "Church St (Hawthorn)": [3001, 4262],
    "Power St": [4263, 4812],
    "Burnley St / Swan St": [4821, 4812],  # Burnley St runs south from Victoria St to Swan St  # Power St runs south from Burwood Rd towards Swan St / Madden Gv
    "Denmark St / Power St": [3662, 3002, 4263],
    "Princess St": [3662, 2820],
    # East-west arterials (listed west to east).
    "Victoria St / Barkers Rd": [4821, 3001, 3002, 4035],  # Victoria St becomes Barkers Rd at the Yarra
    "High St (Kew)": [3001, 3662, 4335, 4321, 4030],
    "Cotham Rd / Whitehorse Rd / Maroondah Hwy": [3662, 4324, 4034, 4063, 2200],
    "Harp Rd / Belmore Rd": [4321, 4032, 4057],
    "Doncaster Rd": [4030, 4051, 3180],
    "Eastern Fwy": [2825, 2827],
    "Bridge Rd / Burwood Rd": [4262, 4263, 4264, 4266, 4040],  # Burwood Rd ends at Camberwell Junction (4040)
    "Canterbury Rd": [3120, 3122, 3127, 3126],
    "Riversdale Rd": [4270, 4272, 4040, 3804, 3682],
    "Camberwell Rd": [4040, 3812],
    "Toorak Rd / Burwood Hwy": [4273, 4043, 2000],
    "High St / High Street Rd (Glen Iris)": [2846, 970],  # High St becomes High Street Rd at Warrigal Rd
}

CONTROLLED_INTERSECTIONS_PER_LINK = 1
EARTH_RADIUS_KM = 6371.0088


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in kilometres between two (lat, lon) points."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def load_site_coordinates(workbook: Path = RAW_WORKBOOK) -> dict[int, tuple[float, float]]:
    """Return {scats_id: (lat, lon)} averaged over each site's detector approaches."""
    import pandas as pd  # Imported here so the rest of the module has no pandas dependency.

    data = pd.read_excel(workbook, sheet_name="Data", header=1)
    approaches = data[["SCATS Number", "NB_LATITUDE", "NB_LONGITUDE"]].drop_duplicates()
    approaches = approaches[(approaches["NB_LATITUDE"] != 0) & (approaches["NB_LONGITUDE"] != 0)]
    means = approaches.groupby("SCATS Number")[["NB_LATITUDE", "NB_LONGITUDE"]].mean()
    return {
        int(site): (float(row.NB_LATITUDE), float(row.NB_LONGITUDE))
        for site, row in means.iterrows()
    }


def build_links(
    coordinates: dict[int, tuple[float, float]],
    segments: dict[str, list[int]] = ROAD_SEGMENTS,
) -> list[tuple[int, int, float, int]]:
    """Turn road segments into sorted directed links (from, to, km, controlled)."""
    links: dict[tuple[int, int], float] = {}
    for road, sites in segments.items():
        for site in sites:
            if site not in coordinates:
                raise ValueError(f"{road}: SCATS site {site} has no coordinates.")
        for a, b in zip(sites, sites[1:]):
            distance = round(haversine_km(coordinates[a], coordinates[b]), 3)
            links[(a, b)] = distance
            links[(b, a)] = distance
    return [
        (a, b, distance, CONTROLLED_INTERSECTIONS_PER_LINK)
        for (a, b), distance in sorted(links.items())
    ]


def unreachable_sites(site_ids: set[int], links: list[tuple[int, int, float, int]]) -> set[int]:
    """Sites that cannot be reached from the first site (the network must be connected)."""
    graph: dict[int, set[int]] = defaultdict(set)
    for a, b, *_ in links:
        graph[a].add(b)
    start = min(site_ids)
    seen = {start}
    queue = deque([start])
    while queue:
        for neighbour in graph[queue.popleft()]:
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)
    return site_ids - seen


def write_network() -> tuple[int, int]:
    """Write scats_sites.csv (with coordinates) and road_links.csv."""
    coordinates = load_site_coordinates()

    with SITES_CSV.open(newline="", encoding="utf-8") as handle:
        sites = list(csv.DictReader(handle))
    site_ids = {int(row["scats_id"]) for row in sites}
    missing = site_ids - set(coordinates)
    if missing:
        raise ValueError(f"No coordinates in the workbook for sites: {sorted(missing)}")

    links = build_links(coordinates)
    disconnected = unreachable_sites(site_ids, links)
    if disconnected:
        raise ValueError(f"Road network is not connected; unreachable: {sorted(disconnected)}")

    with SITES_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scats_id", "site_name", "latitude", "longitude"])
        for row in sites:
            lat, lon = coordinates[int(row["scats_id"])]
            writer.writerow([row["scats_id"], row["site_name"], f"{lat:.6f}", f"{lon:.6f}"])

    with LINKS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["from_scats", "to_scats", "distance_km", "controlled_intersections"])
        writer.writerows(links)

    return len(sites), len(links)


if __name__ == "__main__":
    site_count, link_count = write_network()
    print(f"Wrote {site_count} sites to {SITES_CSV.relative_to(BASE_DIR)}")
    print(f"Wrote {link_count} directed links to {LINKS_CSV.relative_to(BASE_DIR)}")
