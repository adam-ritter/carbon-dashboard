"""
Generate sample sustainability database with realistic operational metrics
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import random

# -----------------------------
# MAIN ENTRY POINT (FIXED)
# -----------------------------

def generate_sustainability_database(db_path):
    """
    Generate database with actual operational metrics
    """
    DB_PATH = Path(db_path)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing DB to ensure clean rebuild
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # -----------------------------
    # CREATE TABLES
    # -----------------------------

    cursor.execute("""
    CREATE TABLE facilities (
        facility_id TEXT PRIMARY KEY,
        facility_name TEXT,
        region TEXT,
        facility_type TEXT,
        operational_start_date DATE
    )
    """)

    cursor.execute("""
    CREATE TABLE emissions_monthly (
        emission_id INTEGER PRIMARY KEY AUTOINCREMENT,
        facility_id TEXT,
        date DATE,
        scope1_tonnes REAL,
        scope2_location_tonnes REAL,
        scope2_market_tonnes REAL,
        scope3_tonnes REAL,
        electricity_mwh REAL,
        renewable_pct REAL,
        FOREIGN KEY (facility_id) REFERENCES facilities (facility_id)
    )
    """)

    cursor.execute("""
    CREATE TABLE facility_operational_metrics (
        metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
        facility_id TEXT,
        date DATE,
        fuel_consumption_mwh REAL,
        renewable_electricity_mwh REAL,
        cfe_pct REAL,
        pue REAL,
        water_withdrawal_gallons REAL,
        water_discharge_gallons REAL,
        water_consumption_gallons REAL,
        water_replenishment_pct REAL,
        waste_generated_tons REAL,
        waste_diverted_tons REAL,
        waste_diversion_pct REAL,
        energy_cost_usd REAL,
        water_cost_usd REAL,
        carbon_cost_usd REAL,
        FOREIGN KEY (facility_id) REFERENCES facilities (facility_id)
    )
    """)

    cursor.execute("""
    CREATE TABLE emission_factors (
        factor_name TEXT,
        scope INTEGER,
        emission_factor REAL,
        unit TEXT,
        source TEXT,
        geography TEXT,
        year INTEGER
    )
    """)

    cursor.execute("""
    CREATE TABLE emission_targets (
        scope INTEGER,
        baseline_year INTEGER,
        baseline_emissions REAL,
        target_year INTEGER,
        reduction_percent REAL,
        target_emissions REAL,
        sbti_aligned BOOLEAN
    )
    """)

    cursor.execute("""
    CREATE TABLE scope3_categories (
        category_id INTEGER,
        category_name TEXT,
        description TEXT,
        avg_emissions_tonnes REAL
    )
    """)

    # -----------------------------
    # INSERT SAMPLE DATA
    # -----------------------------

    facilities = [
        ("DC_US_WEST_1", "US West Data Center", "North America", "Data Center", "2018-01-01"),
        ("DC_US_EAST_1", "US East Data Center", "North America", "Data Center", "2019-06-01"),
        ("DC_EU_1", "EU Central Data Center", "Europe", "Data Center", "2020-03-01"),
        ("MFG_ASIA_1", "Asia Manufacturing Plant", "Asia", "Manufacturing", "2017-09-01"),
        ("OFFICE_GLOBAL", "Global HQ Office", "Global", "Office", "2015-01-01"),
    ]

    cursor.executemany("""
    INSERT INTO facilities VALUES (?, ?, ?, ?, ?)
    """, facilities)

    dates = pd.date_range("2022-01-01", "2024-12-01", freq="MS")

    for facility_id, _, _, facility_type, _ in facilities:
        for date in dates:
            scope1 = random.uniform(50, 300)
            scope2_market = random.uniform(100, 800)
            scope2_location = scope2_market * random.uniform(1.05, 1.25)
            scope3 = random.uniform(200, 1200)

            electricity = scope2_market * random.uniform(0.8, 1.2)
            renewable_pct = random.uniform(30, 90)

            cursor.execute("""
            INSERT INTO emissions_monthly (
                facility_id, date,
                scope1_tonnes, scope2_location_tonnes,
                scope2_market_tonnes, scope3_tonnes,
                electricity_mwh, renewable_pct
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                facility_id, date.strftime("%Y-%m-%d"),
                scope1, scope2_location,
                scope2_market, scope3,
                electricity, renewable_pct
            ))

            fuel = random.uniform(200, 1500)
            renewable_electricity = electricity * renewable_pct / 100
            cfe = renewable_pct
            pue = random.uniform(1.2, 1.6) if facility_type == "Data Center" else None

            water_withdrawal = random.uniform(1e6, 5e6)
            water_discharge = water_withdrawal * random.uniform(0.7, 0.9)
            water_consumption = water_withdrawal - water_discharge
            water_replen = random.uniform(20, 80)

            waste_generated = random.uniform(50, 400)
            waste_diverted = waste_generated * random.uniform(0.5, 0.9)
            waste_diversion = waste_diverted / waste_generated * 100

            energy_cost = electricity * random.uniform(50, 120)
            water_cost = water_withdrawal * 0.002
            carbon_cost = (scope1 + scope2_market + scope3) * random.uniform(30, 80)

            cursor.execute("""
            INSERT INTO facility_operational_metrics VALUES (
                NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """, (
                facility_id, date.strftime("%Y-%m-%d"),
                fuel, renewable_electricity, cfe, pue,
                water_withdrawal, water_discharge,
                water_consumption, water_replen,
                waste_generated, waste_diverted, waste_diversion,
                energy_cost, water_cost, carbon_cost
            ))

    emission_factors = [
        ("Natural Gas", 1, 0.053, "tCO2e/MWh", "EPA", "US", 2023),
        ("Grid Electricity", 2, 0.42, "tCO2e/MWh", "IEA", "Global", 2023),
        ("Business Travel", 3, 0.15, "tCO2e/flight", "DEFRA", "Global", 2023),
    ]

    cursor.executemany("""
    INSERT INTO emission_factors VALUES (?, ?, ?, ?, ?, ?, ?)
    """, emission_factors)

    targets = [
        (1, 2020, 50000, 2030, 50, 25000, True),
        (2, 2020, 80000, 2030, 75, 20000, True),
        (3, 2020, 120000, 2035, 30, 84000, False),
    ]

    cursor.executemany("""
    INSERT INTO emission_targets VALUES (?, ?, ?, ?, ?, ?, ?)
    """, targets)

    scope3_categories = [
        (1, "Purchased Goods", "Upstream supply chain", 30000),
        (6, "Business Travel", "Flights and lodging", 15000),
        (11, "Use of Sold Products", "Customer usage", 50000),
    ]

    cursor.executemany("""
    INSERT INTO scope3_categories VALUES (?, ?, ?, ?)
    """, scope3_categories)

    conn.commit()
    conn.close()

    return DB_PATH
