"""
Data loading utilities for sustainability dashboard
Loads emissions and operational metrics from database
"""

import pandas as pd
import sqlite3
from typing import Optional, List
from pathlib import Path

# ------------------------------------------------------------------
# Canonical database paths (absolute, deterministic)
# ------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]  # utils -> streamlit_app -> repo root
DATA_DIR = REPO_ROOT / "data"


RAW_DB_PATH = DATA_DIR / "sustainability_data.db"
CLEAN_DB_PATH = DATA_DIR / "sustainability_data_clean.db"

# ------------------------------------------------------------------
# Database helpers
# ------------------------------------------------------------------

def get_database_path() -> Path:
    """
    Return cleaned database if it exists, otherwise raw database
    """
    return CLEAN_DB_PATH if CLEAN_DB_PATH.exists() else RAW_DB_PATH


def get_db_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """
    Create a SQLite connection using an absolute path
    """
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
    db_path: Optional[Path] = None
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

    params = []

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


def get_summary_statistics(db_path: Optional[Path] = None) -> dict:
    """
    Get summary statistics for dashboard
    """
    conn = get_db_connection(db_path)
    stats = {}

    stats["scope_totals"] = pd.read_sql_query(
        """
        SELECT 
            SUM(scope1_tonnes) AS scope1,
            SUM(scope2_market_tonnes) AS scope2,
            SUM(scope3_tonnes) AS scope3,
            SUM(scope1_tonnes + scope2_market_tonnes + scope3_tonnes) AS total
        FROM emissions_monthly
        """,
        conn
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
        conn
    ).iloc[0].to_dict()

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
        conn
    ).iloc[0].to_dict()

    conn.close()
    return stats


def get_data_quality_status() -> dict:
    """
    Get current data quality status
    """
    using_cleaned = CLEAN_DB_PATH.exists()
    path = get_database_path()

    return {
        "using_cleaned": using_cleaned,
        "database_path": str(path),
        "database_name": "Cleaned" if using_cleaned else "Raw",
        "status": "✅ Clean" if using_cleaned else "⚠️ Raw (contains quality issues)"
    }
