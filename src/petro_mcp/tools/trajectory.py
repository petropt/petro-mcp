"""Well trajectory and directional survey tools using welleng.

Provides minimum curvature survey calculations, dogleg severity,
vertical section projection, and tortuosity analysis for directional
and horizontal wells.

Reference:
    welleng — https://github.com/jonnymaserati/welleng
    Minimum curvature method per API RP 40 / SPE 84246.
"""

from __future__ import annotations

import json
import math

import numpy as np

try:
    import welleng
    from welleng.survey import Survey, SurveyHeader

    HAS_WELLENG = True
except ImportError:
    HAS_WELLENG = False


def _require_welleng() -> None:
    """Raise a clear error if welleng is not installed."""
    if not HAS_WELLENG:
        raise ImportError(
            "welleng is required for trajectory tools. "
            "Install it with: pip install 'petro-mcp[trajectory]'"
        )


def _safe(value: float) -> float | None:
    """Convert numpy/float to JSON-safe Python float."""
    if value is None:
        return None
    v = float(value)
    if math.isnan(v) or math.isinf(v):
        return None
    return round(v, 6)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_survey_arrays(
    md: list[float],
    inclination: list[float],
    azimuth: list[float],
) -> None:
    """Validate survey station arrays have consistent lengths and values."""
    if len(md) < 2:
        raise ValueError("Need at least 2 survey stations")
    if len(md) != len(inclination) or len(md) != len(azimuth):
        raise ValueError(
            f"Array lengths must match: md={len(md)}, "
            f"inclination={len(inclination)}, azimuth={len(azimuth)}"
        )
    # MD must be monotonically increasing
    for i in range(1, len(md)):
        if md[i] <= md[i - 1]:
            raise ValueError(
                f"MD must be monotonically increasing: "
                f"md[{i - 1}]={md[i - 1]}, md[{i}]={md[i]}"
            )
    # Inclination 0-180, azimuth 0-360
    for i, (inc, azi) in enumerate(zip(inclination, azimuth)):
        if not 0 <= inc <= 180:
            raise ValueError(
                f"Inclination must be 0-180 degrees, got {inc} at station {i}"
            )
        if not 0 <= azi < 360:
            raise ValueError(
                f"Azimuth must be 0-360 degrees, got {azi} at station {i}"
            )


# ---------------------------------------------------------------------------
# 1. Calculate Survey (minimum curvature)
# ---------------------------------------------------------------------------

def calculate_survey(
    md: list[float],
    inclination: list[float],
    azimuth: list[float],
    unit: str = "feet",
) -> str:
    """Calculate well trajectory using the minimum curvature method.

    Takes measured depth, inclination, and azimuth at each survey station
    and returns the computed North, East, TVD, and dogleg severity.

    Args:
        md: List of measured depths (ft or m).
        inclination: List of inclinations (degrees from vertical, 0-180).
        azimuth: List of azimuths (degrees from north, 0-360).
        unit: Depth unit — 'feet' or 'meters'. Default 'feet'.

    Returns:
        JSON string with survey results at each station.
    """
    _require_welleng()
    _validate_survey_arrays(md, inclination, azimuth)

    if unit not in ("feet", "meters"):
        raise ValueError(f"unit must be 'feet' or 'meters', got '{unit}'")

    header = SurveyHeader(depth_unit=unit)
    survey = Survey(
        md=md,
        inc=inclination,
        azi=azimuth,
        deg=True,
        header=header,
        unit=unit,
    )

    stations = []
    for i in range(len(md)):
        stations.append({
            "station": i,
            "md": _safe(survey.md[i]),
            "inclination_deg": _safe(survey.inc_deg[i]),
            "azimuth_deg": _safe(survey.azi_grid_deg[i]),
            "north": _safe(survey.n[i]),
            "east": _safe(survey.e[i]),
            "tvd": _safe(survey.tvd[i]),
            "dls_deg_per_100": _safe(survey.dls[i]),
        })

    # Summary statistics
    total_depth = float(md[-1])
    tvd_at_td = float(survey.tvd[-1])
    max_inc = float(np.max(survey.inc_deg))
    max_dls = float(np.max(survey.dls))
    total_departure = float(
        np.sqrt(survey.n[-1] ** 2 + survey.e[-1] ** 2)
    )

    result = {
        "method": "minimum_curvature",
        "unit": unit,
        "num_stations": len(md),
        "summary": {
            "total_md": _safe(total_depth),
            "tvd_at_td": _safe(tvd_at_td),
            "max_inclination_deg": _safe(max_inc),
            "max_dls_deg_per_100": _safe(max_dls),
            "total_horizontal_departure": _safe(total_departure),
            "north_at_td": _safe(survey.n[-1]),
            "east_at_td": _safe(survey.e[-1]),
        },
        "stations": stations,
    }
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# 2. Dogleg Severity between two stations
# ---------------------------------------------------------------------------

def calculate_dogleg_severity(
    md1: float,
    inc1: float,
    azi1: float,
    md2: float,
    inc2: float,
    azi2: float,
    course_length_unit: str = "feet",
) -> str:
    """Calculate dogleg severity between two survey stations.

    Uses the minimum curvature formula:
        DL = arccos(cos(inc2-inc1) - sin(inc1)*sin(inc2)*(1-cos(azi2-azi1)))
        DLS = DL / (MD2 - MD1) * 100

    Args:
        md1: Measured depth at station 1.
        inc1: Inclination at station 1 (degrees).
        azi1: Azimuth at station 1 (degrees).
        md2: Measured depth at station 2.
        inc2: Inclination at station 2 (degrees).
        azi2: Azimuth at station 2 (degrees).
        course_length_unit: 'feet' or 'meters'. Default 'feet'.

    Returns:
        JSON string with DLS in deg/100ft (or deg/30m).
    """
    _require_welleng()

    if md2 <= md1:
        raise ValueError(f"md2 must be > md1: md1={md1}, md2={md2}")

    header = SurveyHeader(depth_unit=course_length_unit)
    survey = Survey(
        md=[md1, md2],
        inc=[inc1, inc2],
        azi=[azi1, azi2],
        deg=True,
        header=header,
        unit=course_length_unit,
    )

    course_length = md2 - md1
    dogleg_rad = float(survey.dogleg[-1])
    dogleg_deg = math.degrees(dogleg_rad)
    dls = float(survey.dls[-1])

    dls_label = "deg/100ft" if course_length_unit == "feet" else "deg/30m"

    result = {
        "md1": _safe(md1),
        "md2": _safe(md2),
        "course_length": _safe(course_length),
        "dogleg_deg": _safe(dogleg_deg),
        "dogleg_severity": _safe(dls),
        "dls_unit": dls_label,
    }
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# 3. Vertical Section
# ---------------------------------------------------------------------------

def calculate_vertical_section(
    md: list[float],
    inclination: list[float],
    azimuth: list[float],
    vs_azimuth: float = 0.0,
    unit: str = "feet",
) -> str:
    """Project a well trajectory onto a vertical section plane.

    The vertical section is the horizontal displacement projected onto a
    plane defined by the vertical section azimuth. This is the standard
    way to display a well path in a 2D cross-section view.

    Args:
        md: List of measured depths.
        inclination: List of inclinations (degrees).
        azimuth: List of azimuths (degrees).
        vs_azimuth: Vertical section azimuth in degrees (0 = North). Default 0.
        unit: Depth unit — 'feet' or 'meters'. Default 'feet'.

    Returns:
        JSON string with TVD and vertical section displacement at each station.
    """
    _require_welleng()
    _validate_survey_arrays(md, inclination, azimuth)

    if unit not in ("feet", "meters"):
        raise ValueError(f"unit must be 'feet' or 'meters', got '{unit}'")

    header = SurveyHeader(depth_unit=unit)
    survey = Survey(
        md=md,
        inc=inclination,
        azi=azimuth,
        deg=True,
        header=header,
        unit=unit,
    )

    vs = survey.get_vertical_section(vs_azimuth, deg=True)

    stations = []
    for i in range(len(md)):
        stations.append({
            "station": i,
            "md": _safe(survey.md[i]),
            "tvd": _safe(survey.tvd[i]),
            "vertical_section": _safe(vs[i]),
            "north": _safe(survey.n[i]),
            "east": _safe(survey.e[i]),
        })

    result = {
        "vs_azimuth_deg": _safe(vs_azimuth),
        "unit": unit,
        "num_stations": len(md),
        "max_vertical_section": _safe(float(np.max(vs))),
        "stations": stations,
    }
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# 4. Tortuosity Index
# ---------------------------------------------------------------------------

def calculate_tortuosity(
    md: list[float],
    inclination: list[float],
    azimuth: list[float],
    unit: str = "feet",
) -> str:
    """Calculate wellbore tortuosity index from survey data.

    Tortuosity is a measure of how much the wellbore deviates from the
    planned or ideal path. Higher values indicate a more tortuous wellpath,
    which can impact drilling operations, casing running, and production.

    Uses the welleng tortuosity_index() method based on cumulative change
    in well direction relative to course length.

    Args:
        md: List of measured depths.
        inclination: List of inclinations (degrees).
        azimuth: List of azimuths (degrees).
        unit: Depth unit — 'feet' or 'meters'. Default 'feet'.

    Returns:
        JSON string with tortuosity index at each station and summary.
    """
    _require_welleng()
    _validate_survey_arrays(md, inclination, azimuth)

    if unit not in ("feet", "meters"):
        raise ValueError(f"unit must be 'feet' or 'meters', got '{unit}'")

    header = SurveyHeader(depth_unit=unit)
    survey = Survey(
        md=md,
        inc=inclination,
        azi=azimuth,
        deg=True,
        header=header,
        unit=unit,
    )

    ti = survey.tortuosity_index()

    stations = []
    for i in range(len(md)):
        stations.append({
            "station": i,
            "md": _safe(survey.md[i]),
            "inclination_deg": _safe(survey.inc_deg[i]),
            "azimuth_deg": _safe(survey.azi_grid_deg[i]),
            "tortuosity_index": _safe(ti[i]),
            "dls_deg_per_100": _safe(survey.dls[i]),
        })

    result = {
        "unit": unit,
        "num_stations": len(md),
        "summary": {
            "max_tortuosity_index": _safe(float(np.max(ti))),
            "mean_tortuosity_index": _safe(float(np.mean(ti[1:]))),
            "max_dls_deg_per_100": _safe(float(np.max(survey.dls))),
        },
        "stations": stations,
    }
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# 5. Anti-collision check (closest approach between two wells)
# ---------------------------------------------------------------------------

def check_anticollision(
    well1_md: list[float],
    well1_inc: list[float],
    well1_azi: list[float],
    well2_md: list[float],
    well2_inc: list[float],
    well2_azi: list[float],
    well2_start_north: float = 0.0,
    well2_start_east: float = 0.0,
    unit: str = "feet",
) -> str:
    """Check separation between two wells at closest approach.

    Computes the center-to-center distance between two well trajectories
    at each survey station pair and identifies the closest approach point.

    Args:
        well1_md: Measured depths for reference well.
        well1_inc: Inclinations for reference well (degrees).
        well1_azi: Azimuths for reference well (degrees).
        well2_md: Measured depths for offset well.
        well2_inc: Inclinations for offset well (degrees).
        well2_azi: Azimuths for offset well (degrees).
        well2_start_north: Offset well surface location north of reference (ft or m).
        well2_start_east: Offset well surface location east of reference (ft or m).
        unit: Depth unit — 'feet' or 'meters'. Default 'feet'.

    Returns:
        JSON string with closest approach distance and details.
    """
    _require_welleng()
    _validate_survey_arrays(well1_md, well1_inc, well1_azi)
    _validate_survey_arrays(well2_md, well2_inc, well2_azi)

    if unit not in ("feet", "meters"):
        raise ValueError(f"unit must be 'feet' or 'meters', got '{unit}'")

    header = SurveyHeader(depth_unit=unit)

    survey1 = Survey(
        md=well1_md, inc=well1_inc, azi=well1_azi,
        deg=True, header=header, unit=unit,
    )
    survey2 = Survey(
        md=well2_md, inc=well2_inc, azi=well2_azi,
        deg=True, header=header, unit=unit,
        start_nev=[well2_start_north, well2_start_east, 0.0],
    )

    # Compute pairwise distances between all station positions
    pos1 = survey1.pos_nev  # (n1, 3) — north, east, vertical
    pos2 = survey2.pos_nev  # (n2, 3)

    # Distance matrix
    from scipy.spatial.distance import cdist

    dist_matrix = cdist(pos1, pos2)
    min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
    min_distance = float(dist_matrix[min_idx])

    i_ref, i_off = int(min_idx[0]), int(min_idx[1])

    result = {
        "unit": unit,
        "closest_approach": {
            "distance": _safe(min_distance),
            "reference_well_md": _safe(survey1.md[i_ref]),
            "reference_well_tvd": _safe(survey1.tvd[i_ref]),
            "offset_well_md": _safe(survey2.md[i_off]),
            "offset_well_tvd": _safe(survey2.tvd[i_off]),
            "reference_well_station": i_ref,
            "offset_well_station": i_off,
        },
        "separation_at_td": {
            "reference_td": _safe(float(well1_md[-1])),
            "offset_td": _safe(float(well2_md[-1])),
            "distance_at_ref_td": _safe(float(np.min(dist_matrix[-1, :]))),
            "distance_at_off_td": _safe(float(np.min(dist_matrix[:, -1]))),
        },
    }
    return json.dumps(result, indent=2)
