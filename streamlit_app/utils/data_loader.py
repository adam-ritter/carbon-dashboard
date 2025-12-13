"""
Data loading utilities for sustainability dashboard
Loads emissions and operational metrics from database
"""

from __future__ import annotations

import pandas as pd
import sqlite3
from typing import Optional, List
from pathlib import Path

# ------------------------------------------------------------------
# Canonical database paths (absolute, deterministic)
# Repo root: utils -> streamlit_app -> repo root
# ------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

RAW_DB_PATH = DATA_DIR / "sustainability_data.db"
CLEAN_DB_PATH = DATA_DIR / "sustainability_data_clean.db"

# ------------------------------------------------------------------
# Database helpers
# ------------------------------------------------------------------

def get_database_path() -> Path:
    """Return cleaned database if it exists, otherwise raw database."""
    return CLEAN_DB_PATH if CLEAN_DB_PATH.exists() else RAW_DB_PATH


def get_db_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Create a SQLite connection using an absolute path."""
    path = Path(db_path) if db_path else get_database_path()

    if not path.exists():
        raise FileNotFoundError(f"Database not found at {path}")

    return sqlite3.connect(path, check_same_thread=False)

# ------------------------------------------------------------------
# Loaders
# ------------------------------------------------------------------

def load_emissions_data(
    facility_ids: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    conn = get_db_connection(db_path)

    query = """
    SELECT 
        e.emission_id,
        e.facility_id,
        f.facility_name,
        f.region,
        f.facility_type,
        e.date,
        e.scope1_tonnes,
        e.scope2_location_tonnes,
        e.scope2_market_tonnes,
        e.scope3_tonnes,
        (e.scope1_tonnes + e.scope2_market_tonnes + e.scope3_tonnes) AS total_emissions,
        e.electricity_mwh,
        e.renewable_pct
    FROM emissions_monthly e
    JOIN facilities f ON e.facility_id = f.facility_id
    WHERE 1=1
    """

    params: List[object] = []

    if facility_ids:
        placeholders = ",".join("?" * len(facility_ids))
        query += f" AND e.facility_id IN ({placeholders})"
        params.extend(facility_ids)

    if start_date:
        query += " AND e.date >= ?"
        params.append(start_date)

    if end_date:
        query += " AND e.date <= ?"
        params.append(end_date)

    query += " ORDER BY e.date, e.facility_id"

    df = pd.read_sql_query(query, conn, params=params)
    df["date"] = pd.to_datetime(df["date"])
    conn.close()
    return df


def load_operational_metrics(
    facility_ids: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    conn = get_db_connection(db_path)

    query = """
    SELECT 
        o.metric_id,
        o.facility_id,
        f.facility_name,
        f.region,
        f.facility_type,
        o.date,
        o.fuel_consumption_mwh,
        o.renewable_electricity_mwh,
        o.cfe_pct,
        o.pue,
        o.water_withdrawal_gallons,
        o.water_discharge_gallons,
        o.water_consumption_gallons,
        o.water_replenishment_pct,
        o.waste_generated_tons,
        o.waste_diverted_tons,
        o.waste_diversion_pct,
        o.energy_cost_usd,
        o.water_cost_usd,
        o.carbon_cost_usd
    FROM facility_operational_metrics o
    JOIN facilities f ON o.facility_id = f.facility_id
    WHERE 1=1
    """

    params: List[object] = []

    if facility_ids:
        placeholders = ",".join("?" * len(facility_ids))
        query += f" AND o.facility_id IN ({placeholders})"
        params.extend(facility_ids)

    if start_date:
        query += " AND o.date >= ?"
        params.append(start_date)

    if end_date:
        query += " AND o.date <= ?"
        params.append(end_date)

    query += " ORDER BY o.date, o.facility_id"

    df = pd.read_sql_query(query, conn, params=params)
    df["date"] = pd.to_datetime(df["date"])
    conn.close()
    return df


def load_combined_metrics(
    facility_ids: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    conn = get_db_connection(db_path)

    query = """
    SELECT 
        e.facility_id,
        f.facility_name,
        f.region,
        f.facility_type,
        e.date,
        e.scope1_tonnes,
        e.scope2_location_tonnes,
        e.scope2_market_tonnes,
        e.scope3_tonnes,
        (e.scope1_tonnes + e.scope2_market_tonnes + e.scope3_tonnes) AS total_emissions,
        e.electricity_mwh,
        e.renewable_pct,
        o.fuel_consumption_mwh,
        o.renewable_electricity_mwh,
        o.cfe_pct,
        o.pue,
        o.water_withdrawal_gallons,
        o.water_discharge_gallons,
        o.water_consumption_gallons,
        o.water_replenishment_pct,
        o.waste_generated_tons,
        o.waste_diverted_tons,
        o.waste_diversion_pct,
        o.energy_cost_usd,
        o.water_cost_usd,
        o.carbon_cost_usd
    FROM emissions_monthly e
    JOIN facilities f ON e.facility_id = f.facility_id
    LEFT JOIN facility_operational_metrics o
        ON e.facility_id = o.facility_id AND e.date = o.date
    WHERE 1=1
    """

    params: List[object] = []

    if facility_ids:
        placeholders = ",".join("?" * len(facility_ids))
        query += f" AND e.facility_id IN ({placeholders})"
        params.extend(facility_ids)

    if start_date:
        query += " AND e.date >= ?"
        params.append(start_date)

    if end_date:
        query += " AND e.date <= ?"
        params.append(end_date)

    query += " ORDER BY e.date, e.facility_id"

    df = pd.read_sql_query(query, conn, params=params)
    df["date"] = pd.to_datetime(df["date"])
    conn.close()
    return df


def load_facilities(db_path: Optional[Path] = None) -> pd.DataFrame:
    conn = get_db_connection(db_path)

    query = """
    SELECT 
        facility_id,
        facility_name,
        region,
        facility_type,
        operational_start_date
    FROM facilities
    ORDER BY facility_name
    """

    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def load_emission_factors(db_path: Optional[Path] = None) -> pd.DataFrame:
    conn = get_db_connection(db_path)

    query = """
    SELECT 
        factor_name,
        scope,
        emission_factor,
        unit,
        source,
        geography,
        year,
        last_updated
    FROM emission_factors
    ORDER BY scope, factor_name
    """

    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def load_targets(db_path: Optional[Path] = None) -> pd.DataFrame:
    conn = get_db_connection(db_path)

    query = """
    SELECT 
        scope,
        baseline_year,
        baseline_emissions,
        target_year,
        reduction_percent,
        target_emissions,
        sbti_aligned
    FROM emission_targets
    ORDER BY scope
    """

    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_summary_statistics(db_path: Optional[Path] = None) -> dict:
    conn = get_db_connection(db_path)
    stats: dict = {}

    stats["scope_totals"] = pd.read_sql_query(
        """
        SELECT 
            SUM(scope1_tonnes) AS scope1,
            SUM(scope2_market_tonnes) AS scope2,
            SUM(scope3_tonnes) AS scope3,
            SUM(scope1_tonnes + scope2_market_tonnes + scope3_tonnes) AS total
        FROM emissions_monthly
        """,
        conn,
    ).iloc[0].to_dict()

    stats["recent_month"] = pd.read_sql_query(
        """
        SELECT 
            date,
            SUM(scope1_tonnes + scope2_market_tonnes + scope3_tonnes) AS total_emissions
        FROM emissions_monthly
        GROUP BY date
        ORDER BY date DESC
        LIMIT 1
        """,
        conn,
    ).iloc[0].to_dict()

    # Optional YoY calc (if your UI expects it)
    yoy = pd.read_sql_query(
        """
        WITH recent AS (
            SELECT date, SUM(scope1_tonnes + scope2_market_tonnes + scope3_tonnes) AS emissions
            FROM emissions_monthly
            GROUP BY date
            ORDER BY date DESC
            LIMIT 1
        ),
        prior_year AS (
            SELECT date, SUM(scope1_tonnes + scope2_market_tonnes + scope3_tonnes) AS emissions
            FROM emissions_monthly
            WHERE date = (SELECT date(date, '-1 year') FROM recent)
            GROUP BY date
        )
        SELECT 
            r.emissions AS current,
            p.emissions AS prior,
            ROUND((r.emissions - p.emissions) / p.emissions * 100, 1) AS pct_change
        FROM recent r, prior_year p
        """,
        conn,
    )
    if len(yoy) > 0:
        stats["yoy_change"] = yoy.iloc[0].to_dict()

    stats["operational"] = pd.read_sql_query(
        """
        SELECT 
            AVG(pue) AS avg_pue,
            AVG(cfe_pct) AS avg_cfe,
            AVG(water_replenishment_pct) AS avg_water_replen,
            AVG(waste_diversion_pct) AS avg_waste_diversion
        FROM facility_operational_metrics
        WHERE pue IS NOT NULL
        """,
        conn,
    ).iloc[0].to_dict()

    conn.close()
    return stats


def get_data_quality_status() -> dict:
    using_cleaned = CLEAN_DB_PATH.exists()
    path = get_database_path()
    return {
        "using_cleaned": using_cleaned,
        "database_path": str(path),
        "database_name": "Cleaned" if using_cleaned else "Raw",
        "status": "✅ Clean" if using_cleaned else "⚠️ Raw (contains quality issues)",
    }
