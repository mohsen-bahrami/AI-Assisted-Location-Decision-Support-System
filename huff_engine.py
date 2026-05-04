import math
import time
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).parent / "Data"


def _haversine_miles(lat1, lon1, lat2, lon2):
    r = 3958.8
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlambda = math.radians(float(lon2) - float(lon1))

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def run_huff_model(candidate_lat, candidate_lon, business_category, floor_area, db_connection=None):
    """
    Baseline interface for all teams.

    Teams may completely rewrite the internals, but should keep this function name,
    parameters, and return structure.

    Required inputs:
        candidate_lat: float
        candidate_lon: float
        business_category: str
        floor_area: float
        db_connection: optional Azure SQL connection

    Required output:
        dict with predicted_visits, market_share, competitors, runtime_ms, notes
    """

    start = time.perf_counter()

    candidate_lat = float(candidate_lat)
    candidate_lon = float(candidate_lon)
    floor_area = float(floor_area)

    pois_path = DATA_DIR / "worcester_pois.csv"
    params_path = DATA_DIR / "calibrated_parameters_filtered.csv"
    visits_path = DATA_DIR / "worcester_cbg_poi_visits.csv"

    pois = pd.read_csv(pois_path)
    params = pd.read_csv(params_path)
    visits = pd.read_csv(visits_path)

    matching = pois[
        pois["top_category"].astype(str).str.lower()
        == str(business_category).strip().lower()
    ].copy()

    if matching.empty:
        matching = pois.copy()
        category_note = (
            f"No exact POI category match found for '{business_category}'. "
            "Baseline used all POIs as competitors."
        )
    else:
        category_note = f"Used competitors from category '{business_category}'."

    matching["distance_miles"] = matching.apply(
        lambda row: _haversine_miles(
            candidate_lat,
            candidate_lon,
            row["latitude"],
            row["longitude"],
        ),
        axis=1,
    )

    nearby = matching.sort_values("distance_miles").head(10).copy()

    param_row = params[
        params["top_category"].astype(str).str.lower()
        == str(business_category).strip().lower()
    ]

    if not param_row.empty:
        alpha = float(param_row.iloc[0]["alpha"])
        beta = float(param_row.iloc[0]["beta"])
    else:
        alpha = 1.5
        beta = 1.0

    candidate_attraction = max(floor_area, 1.0) ** alpha

    competitor_attractions = []
    for _, row in nearby.iterrows():
        comp_size = row.get("wkt_area_sq_meters", 100)
        try:
            comp_size = float(comp_size)
        except Exception:
            comp_size = 100

        distance = max(float(row["distance_miles"]), 0.05)
        attraction = (max(comp_size, 1.0) ** alpha) / (distance ** beta)

        competitor_attractions.append(
            {
                "name": str(row.get("location_name", "Unknown")),
                "category": str(row.get("top_category", "")),
                "lat": float(row["latitude"]),
                "lon": float(row["longitude"]),
                "distance_miles": round(distance, 3),
                "size": round(comp_size, 2),
                "attraction": round(attraction, 4),
            }
        )

    candidate_effective_attraction = candidate_attraction / (0.25 ** beta)
    total_attraction = candidate_effective_attraction + sum(
        c["attraction"] for c in competitor_attractions
    )

    market_share = (
        candidate_effective_attraction / total_attraction
        if total_attraction > 0
        else 0
    )

    total_observed_visits = int(visits["visit_count"].sum())
    baseline_market_size = max(total_observed_visits / 52, 1)

    predicted_visits = baseline_market_size * market_share

    runtime_ms = round((time.perf_counter() - start) * 1000, 2)

    return {
        "predicted_visits": round(float(predicted_visits), 2),
        "market_share": round(float(market_share), 4),
        "competitors": competitor_attractions,
        "runtime_ms": runtime_ms,
        "notes": (
            "This is a simple baseline Huff-style calculation. "
            "Teams should replace the internals with their optimized model and Azure SQL logic. "
            + category_note
        ),
        "inputs": {
            "candidate_lat": candidate_lat,
            "candidate_lon": candidate_lon,
            "business_category": business_category,
            "floor_area": floor_area,
            "alpha": alpha,
            "beta": beta,
        },
    }
