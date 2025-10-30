"""
Data loading utilities for sustainability dashboard
Automatically uses cleaned database if available
"""

import pandas as pd
import sqlite3
import os
from typing import Optional, List, Tuple
from datetime import datetime

def get_database_path():
    """
    Return path to cleaned database if it exists, otherwise raw database
    This allows seamless switching between raw and cleaned data
    """
    cleaned_path = '../data/sustainability_data_clean.db'
    raw_path = '../data/sustainability_data.db'
    
    if os.path.exists(cleaned_path):
        return cleaned_path
    else:
        return raw_path

def get_db_connection(db_path: Optional[str] = None):
    """
    Create database connection
    If no path specified, uses cleaned DB if available, otherwise raw DB
    """
    if db_path is None:
        db_path = get_database_path()
    return sqlite3.connect(db_path)

def load_emissions_data(
    facility_ids: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Load emissions data with optional filters
    Automatically uses cleaned database if available
    
    Args:
        facility_ids: List of facility IDs to filter
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        db_path: Path to database (if None, uses cleaned DB if available)
        
    Returns:
        DataFrame with emissions data
    """
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
        (e.scope1_tonnes + e.scope2_market_tonnes + e.scope3_tonnes) as total_emissions,
        e.electricity_mwh,
        e.renewable_pct
    FROM emissions_monthly e
    JOIN facilities f ON e.facility_id = f.facility_id
    WHERE 1=1
    """
    
    params = []
    
    if facility_ids:
        placeholders = ','.join('?' * len(facility_ids))
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
    df['date'] = pd.to_datetime(df['date'])
    
    conn.close()
    
    return df

def load_business_metrics(
    facility_ids: Optional[List[str]] = None,
    db_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Load business metrics data
    Automatically uses cleaned database if available
    """
    conn = get_db_connection(db_path)
    
    query = """
    SELECT 
        b.facility_id,
        f.facility_name,
        b.date,
        b.revenue_millions,
        b.headcount,
        b.square_feet,
        b.server_count,
        b.production_volume
    FROM business_metrics b
    JOIN facilities f ON b.facility_id = f.facility_id
    WHERE 1=1
    """
    
    params = []
    
    if facility_ids:
        placeholders = ','.join('?' * len(facility_ids))
        query += f" AND b.facility_id IN ({placeholders})"
        params.extend(facility_ids)
    
    query += " ORDER BY b.date, b.facility_id"
    
    df = pd.read_sql_query(query, conn, params=params)
    df['date'] = pd.to_datetime(df['date'])
    
    conn.close()
    
    return df

def load_facilities(db_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load facility information
    Automatically uses cleaned database if available
    """
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

def load_emission_factors(db_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load emission factors
    Automatically uses cleaned database if available
    """
    conn = get_db_connection(db_path)
    
    query = """
    SELECT 
        factor_name,
        scope,
        emission_factor,
        unit,
        source,
        geography,
        year
    FROM emission_factors
    ORDER BY scope, factor_name
    """
    
    df = pd.read_sql_query(query, conn)
    
    conn.close()
    
    return df

def load_targets(db_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load emission targets
    Automatically uses cleaned database if available
    """
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

def calculate_emission_intensity(
    emissions: pd.DataFrame,
    business_metrics: pd.DataFrame,
    intensity_type: str = 'revenue'
) -> pd.DataFrame:
    """
    Calculate emission intensity metrics
    
    Args:
        emissions: Emissions DataFrame
        business_metrics: Business metrics DataFrame
        intensity_type: 'revenue', 'headcount', 'square_feet'
        
    Returns:
        DataFrame with intensity metrics
    """
    merged = emissions.merge(
        business_metrics,
        on=['facility_id', 'date'],
        how='inner'
    )
    
    if intensity_type == 'revenue':
        merged['intensity'] = merged['total_emissions'] / merged['revenue_millions']
        unit = 'tonnes CO₂e / $M revenue'
    elif intensity_type == 'headcount':
        merged['intensity'] = merged['total_emissions'] / merged['headcount']
        unit = 'tonnes CO₂e / employee'
    elif intensity_type == 'square_feet':
        merged['intensity'] = merged['total_emissions'] / (merged['square_feet'] / 1000)
        unit = 'tonnes CO₂e / 1000 sqft'
    else:
        raise ValueError(f"Unknown intensity type: {intensity_type}")
    
    merged['intensity_unit'] = unit
    
    return merged

def get_summary_statistics(db_path: Optional[str] = None) -> dict:
    """
    Get summary statistics for dashboard
    Automatically uses cleaned database if available
    """
    conn = get_db_connection(db_path)
    
    stats = {}
    
    # Total emissions by scope
    scope_totals = pd.read_sql_query("""
    SELECT 
        SUM(scope1_tonnes) as scope1,
        SUM(scope2_market_tonnes) as scope2,
        SUM(scope3_tonnes) as scope3,
        SUM(scope1_tonnes + scope2_market_tonnes + scope3_tonnes) as total
    FROM emissions_monthly
    """, conn).iloc[0].to_dict()
    
    stats['scope_totals'] = scope_totals
    
    # Recent month
    recent = pd.read_sql_query("""
    SELECT 
        date,
        SUM(scope1_tonnes + scope2_market_tonnes + scope3_tonnes) as total_emissions
    FROM emissions_monthly
    GROUP BY date
    ORDER BY date DESC
    LIMIT 1
    """, conn).iloc[0].to_dict()
    
    stats['recent_month'] = recent
    
    # YoY change
    yoy = pd.read_sql_query("""
    WITH recent AS (
        SELECT date, SUM(scope1_tonnes + scope2_market_tonnes + scope3_tonnes) as emissions
        FROM emissions_monthly
        GROUP BY date
        ORDER BY date DESC
        LIMIT 1
    ),
    prior_year AS (
        SELECT date, SUM(scope1_tonnes + scope2_market_tonnes + scope3_tonnes) as emissions
        FROM emissions_monthly
        WHERE date = (SELECT date(date, '-1 year') FROM recent)
        GROUP BY date
    )
    SELECT 
        r.emissions as current,
        p.emissions as prior,
        ROUND((r.emissions - p.emissions) / p.emissions * 100, 1) as pct_change
    FROM recent r, prior_year p
    """, conn)
    
    if len(yoy) > 0:
        stats['yoy_change'] = yoy.iloc[0].to_dict()
    
    conn.close()
    
    return stats

def is_using_cleaned_data() -> bool:
    """Check if currently using cleaned database"""
    return os.path.exists('../data/sustainability_data_clean.db')

def get_data_quality_status() -> dict:
    """
    Get current data quality status
    Returns information about which database is being used
    """
    using_cleaned = is_using_cleaned_data()
    current_path = get_database_path()
    
    return {
        'using_cleaned': using_cleaned,
        'database_path': current_path,
        'database_name': 'Cleaned' if using_cleaned else 'Raw',
        'status': '✅ Clean' if using_cleaned else '⚠️ Raw (contains quality issues)'
    }