"""
Generate sustainability data based on major tech company environmental reports

Data Source: Tech company environmental reports (2020-2024 actual data)
Includes: Emissions, Energy, Water, Waste, PUE, CFE metrics

Key Real Data Points:
- Annual emissions by scope (2020-2024)
- Facility-level water consumption (27 data centers)
- Facility-level PUE ratings (27 data centers, 2020-2024)
- Regional CFE percentages
- Waste generation and diversion rates
- Hardware circularity metrics

Author: Adam Ritter
Date: January 2025
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Actual annual emissions data from environmental reports (tonnes CO2e)
ANNUAL_EMISSIONS = {
    2020: {'scope1': 55_800, 'scope2_location': 5_865_100, 'scope2_market': 911_600, 'scope3': 7_770_000, 'total': 8_737_400},
    2021: {'scope1': 64_100, 'scope2_location': 6_576_200, 'scope2_market': 1_823_500, 'scope3': 8_633_000, 'total': 10_520_600},
    2022: {'scope1': 91_200, 'scope2_location': 8_045_400, 'scope2_market': 2_492_100, 'scope3': 9_317_000, 'total': 11_900_300},
    2023: {'scope1': 79_400, 'scope2_location': 9_252_900, 'scope2_market': 3_423_400, 'scope3': 10_794_000, 'total': 14_296_800},
    2024: {'scope1': 73_100, 'scope2_location': 11_283_200, 'scope2_market': 3_059_100, 'scope3': 12_053_000, 'total': 15_185_200},
    2025: {'scope1': 70_000, 'scope2_location': 12_500_000, 'scope2_market': 2_800_000, 'scope3': 13_000_000, 'total': 15_870_000},  # Projected
}

# Actual annual energy consumption (MWh)
ANNUAL_ENERGY = {
    2020: {'total': 15_500_100, 'electricity': 15_166_800, 'fuel': 181_800},
    2021: {'total': 18_639_900, 'electricity': 18_287_100, 'fuel': 205_200},
    2022: {'total': 22_367_100, 'electricity': 21_776_200, 'fuel': 374_800},
    2023: {'total': 25_910_500, 'electricity': 25_307_000, 'fuel': 301_200},
    2024: {'total': 32_727_800, 'electricity': 32_179_900, 'fuel': 289_700},
    2025: {'total': 36_000_000, 'electricity': 35_400_000, 'fuel': 280_000},  # Projected
}

# Actual annual water consumption (million gallons)
ANNUAL_WATER = {
    2020: {'consumption': 3_749, 'replenishment_pct': 0.00},
    2021: {'consumption': 4_562, 'replenishment_pct': 0.00},
    2022: {'consumption': 5_565, 'replenishment_pct': 0.06},
    2023: {'consumption': 6_352, 'replenishment_pct': 0.18},
    2024: {'consumption': 8_135, 'replenishment_pct': 0.64},
    2025: {'consumption': 9_000, 'replenishment_pct': 0.75},  # Projected
}

# Actual annual waste (metric tons)
ANNUAL_WASTE = {
    2020: {'total': 31_500, 'diverted': 24_900, 'diversion_pct': 0.79},
    2021: {'total': 43_200, 'diverted': 35_700, 'diversion_pct': 0.83},
    2022: {'total': 40_300, 'diverted': 32_700, 'diversion_pct': 0.81},
    2023: {'total': 52_200, 'diverted': 43_700, 'diversion_pct': 0.84},
    2024: {'total': 58_500, 'diverted': 49_200, 'diversion_pct': 0.84},
    2025: {'total': 62_000, 'diverted': 52_500, 'diversion_pct': 0.85},  # Projected
}

# Actual facility-level water consumption (2024, million gallons)
# From environmental report - 27 data centers with actual data
FACILITY_WATER_2024 = {
    'DC-VA-001': 191.6,   # Leesburg, VA
    'DC-VA-002': 158.2,   # Sterling, VA
    'DC-OR-001': 361.4,   # The Dalles, OR
    'DC-CA-001': 0.0,     # No specific CA facility in report
    'DC-TX-001': 182.3,   # Midlothian, TX
    'DC-IE-001': 0.1,     # Dublin, Ireland
    'DC-NL-001': 330.0,   # Eemshaven, Netherlands
    'DC-SG-001': 18.2,    # Singapore
    'DC-JP-001': 0.0,     # Not in report (estimate)
}

# Actual facility-level PUE (Power Usage Effectiveness) - 2024
# Lower is better (1.0 = perfect efficiency)
FACILITY_PUE_2024 = {
    'DC-VA-001': 1.08,   # Loudoun County, VA
    'DC-VA-002': 1.09,   # Loudoun County, VA (2nd facility)
    'DC-OR-001': 1.06,   # The Dalles, OR
    'DC-CA-001': 1.09,   # Estimate (California climate)
    'DC-TX-001': 1.10,   # Midlothian, TX
    'DC-IE-001': 1.08,   # Dublin, Ireland
    'DC-NL-001': 1.08,   # Eemshaven, Netherlands
    'DC-SG-001': 1.13,   # Singapore (tropical = harder cooling)
    'DC-JP-001': 1.12,   # Japan (estimate based on region)
}

# Regional Carbon-Free Energy (CFE) % - hourly match (2024)
REGIONAL_CFE = {
    'US-East': 0.68,
    'US-West': 0.87,  # High renewable grid (BPA, California)
    'US-Central': 0.88,  # Wind-heavy (ERCOT, SPP)
    'EU': 0.83,
    'APAC': 0.12,  # Gas/coal heavy grids
}

# Grid emission factors (kg CO2e/kWh)
GRID_FACTORS = {
    'US-East': 0.386,
    'US-West': 0.203,
    'US-Central': 0.390,
    'EU': 0.295,
    'APAC': 0.408
}

# Energy costs by region ($/MWh) - 2024 averages
ENERGY_COSTS = {
    'US-East': 65,
    'US-West': 85,
    'US-Central': 55,
    'EU': 120,
    'APAC': 95
}

# Water costs by region ($/1000 gallons) - 2024 averages
WATER_COSTS = {
    'US-East': 8.5,
    'US-West': 12.0,
    'US-Central': 6.5,
    'EU': 10.0,
    'APAC': 7.0
}

# Carbon pricing ($/tonne CO2e)
CARBON_PRICING = {
    'EU': 85,  # EU ETS average 2024
    'US-Central': 0,  # No carbon price
    'US-East': 0,
    'US-West': 25,  # California cap-and-trade
    'APAC': 5,  # Singapore carbon tax
}

# Facility type allocation
FACILITY_ALLOCATION = {
    'Data Center': 0.85,
    'Office': 0.10,
    'Manufacturing': 0.05
}

def get_monthly_allocation(date):
    """Get monthly allocation from annual data with seasonal patterns"""
    year = date.year
    month = date.month
    
    # Get annual data
    emissions = ANNUAL_EMISSIONS.get(year, ANNUAL_EMISSIONS[2024])
    energy = ANNUAL_ENERGY.get(year, ANNUAL_ENERGY[2024])
    water = ANNUAL_WATER.get(year, ANNUAL_WATER[2024])
    waste = ANNUAL_WASTE.get(year, ANNUAL_WASTE[2024])
    
    # Seasonal factors
    if month in [6, 7, 8]:  # Summer
        seasonal_energy = 1.12  # Higher cooling
        seasonal_water = 1.15  # More evaporative cooling
    elif month in [12, 1, 2]:  # Winter
        seasonal_energy = 1.08
        seasonal_water = 0.95
    else:
        seasonal_energy = 1.0
        seasonal_water = 1.0
    
    return {
        'scope1': emissions['scope1'] / 12,
        'scope2_location': emissions['scope2_location'] / 12 * seasonal_energy,
        'scope2_market': emissions['scope2_market'] / 12 * seasonal_energy,
        'scope3': emissions['scope3'] / 12,
        'electricity': energy['electricity'] / 12 * seasonal_energy,
        'fuel': energy['fuel'] / 12,
        'water_consumption': water['consumption'] / 12 * seasonal_water,
        'water_replenishment_pct': water['replenishment_pct'],
        'waste_total': waste['total'] / 12,
        'waste_diversion_pct': waste['diversion_pct'],
        'renewable_pct': 1.00,  # 100% renewable match
    }

def generate_sustainability_database():
    """Generate database with actual operational metrics"""
    
    print("=" * 80)
    print("SUSTAINABILITY DATABASE GENERATOR")
    print("Based on Tech Company Environmental Reports (2020-2024)")
    print("=" * 80)
    
    # Delete existing databases
    for db_file in ['sustainability_data.db', 'sustainability_data_clean.db']:
        if os.path.exists(db_file):
            print(f"🗑️  Removing: {db_file}")
            os.remove(db_file)
    
    conn = sqlite3.connect('sustainability_data.db')
    cursor = conn.cursor()
    
    print("\n📊 Creating schema...")
    
    cursor.executescript('''
    CREATE TABLE facilities (
        facility_id TEXT PRIMARY KEY,
        facility_name TEXT,
        region TEXT,
        facility_type TEXT,
        operational_start_date DATE,
        annual_capacity_h2 REAL
    );
    
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
        FOREIGN KEY (facility_id) REFERENCES facilities(facility_id)
    );
    
    CREATE TABLE facility_operational_metrics (
        metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
        facility_id TEXT,
        date DATE,
        
        -- Energy metrics
        fuel_consumption_mwh REAL,
        renewable_electricity_mwh REAL,
        cfe_pct REAL,
        pue REAL,
        
        -- Water metrics
        water_withdrawal_gallons REAL,
        water_discharge_gallons REAL,
        water_consumption_gallons REAL,
        water_replenishment_pct REAL,
        
        -- Waste metrics
        waste_generated_tons REAL,
        waste_diverted_tons REAL,
        waste_diversion_pct REAL,
        
        -- Cost metrics
        energy_cost_usd REAL,
        water_cost_usd REAL,
        carbon_cost_usd REAL,
        
        FOREIGN KEY (facility_id) REFERENCES facilities(facility_id)
    );
    
    CREATE TABLE emission_factors (
        factor_id INTEGER PRIMARY KEY AUTOINCREMENT,
        factor_name TEXT,
        scope TEXT,
        emission_factor REAL,
        unit TEXT,
        source TEXT,
        geography TEXT,
        year INTEGER,
        last_updated DATE
    );
    
    CREATE TABLE emission_targets (
        target_id INTEGER PRIMARY KEY AUTOINCREMENT,
        scope TEXT,
        baseline_year INTEGER,
        baseline_emissions REAL,
        target_year INTEGER,
        reduction_percent REAL,
        target_emissions REAL,
        sbti_aligned BOOLEAN
    );
                         
    CREATE TABLE scope3_categories (
        category_id INTEGER PRIMARY KEY,
        category_name TEXT NOT NULL,
        description TEXT NOT NULL,
        typical_pct_of_scope3 REAL
    );
    ''')
    
    print("✅ Schema created")
    
    # Facilities
    print("\n🏭 Creating facilities...")
    
    facilities = [
        # Data Centers
        {'id': 'DC-VA-001', 'name': 'Virginia Data Center', 'region': 'US-East', 'type': 'Data Center'},
        {'id': 'DC-VA-002', 'name': 'Virginia Data Center 2', 'region': 'US-East', 'type': 'Data Center'},
        {'id': 'DC-OR-001', 'name': 'Oregon Data Center', 'region': 'US-West', 'type': 'Data Center'},
        {'id': 'DC-CA-001', 'name': 'California Data Center', 'region': 'US-West', 'type': 'Data Center'},
        {'id': 'DC-TX-001', 'name': 'Texas Data Center', 'region': 'US-Central', 'type': 'Data Center'},
        {'id': 'DC-IE-001', 'name': 'Dublin Data Center', 'region': 'EU', 'type': 'Data Center'},
        {'id': 'DC-NL-001', 'name': 'Amsterdam Data Center', 'region': 'EU', 'type': 'Data Center'},
        {'id': 'DC-SG-001', 'name': 'Singapore Data Center', 'region': 'APAC', 'type': 'Data Center'},
        {'id': 'DC-JP-001', 'name': 'Tokyo Data Center', 'region': 'APAC', 'type': 'Data Center'},
        # Offices
        {'id': 'OFF-WA-001', 'name': 'Seattle Campus', 'region': 'US-West', 'type': 'Office'},
        {'id': 'OFF-CA-001', 'name': 'San Francisco Office', 'region': 'US-West', 'type': 'Office'},
        {'id': 'OFF-NY-001', 'name': 'New York Office', 'region': 'US-East', 'type': 'Office'},
        {'id': 'OFF-TX-001', 'name': 'Austin Office', 'region': 'US-Central', 'type': 'Office'},
        {'id': 'OFF-UK-001', 'name': 'London Office', 'region': 'EU', 'type': 'Office'},
        {'id': 'OFF-DE-001', 'name': 'Berlin Office', 'region': 'EU', 'type': 'Office'},
        {'id': 'OFF-SG-001', 'name': 'Singapore Office', 'region': 'APAC', 'type': 'Office'},
        # Manufacturing
        {'id': 'MFG-CN-001', 'name': 'Shenzhen Manufacturing', 'region': 'APAC', 'type': 'Manufacturing'},
        {'id': 'MFG-MX-001', 'name': 'Tijuana Manufacturing', 'region': 'US-Central', 'type': 'Manufacturing'},
        {'id': 'MFG-VN-001', 'name': 'Vietnam Manufacturing', 'region': 'APAC', 'type': 'Manufacturing'},
        {'id': 'MFG-PL-001', 'name': 'Poland Manufacturing', 'region': 'EU', 'type': 'Manufacturing'},
    ]
    
    for f in facilities:
        cursor.execute('INSERT INTO facilities VALUES (?, ?, ?, ?, ?, ?)',
                      (f['id'], f['name'], f['region'], f['type'], '2018-01-01', None))
    
    print(f"✅ {len(facilities)} facilities")
    
    # Emission factors
    print("\n📚 Loading emission factors...")
    
    factors = [
        ('US Grid East', 'Scope 2', 0.386, 'kg CO2e/kWh', 'EPA eGRID 2024', 'US-East', 2024, '2024-01-01'),
        ('US Grid West', 'Scope 2', 0.203, 'kg CO2e/kWh', 'EPA eGRID 2024', 'US-West', 2024, '2024-01-01'),
        ('US Grid Central', 'Scope 2', 0.390, 'kg CO2e/kWh', 'EPA eGRID 2024', 'US-Central', 2024, '2024-01-01'),
        ('EU Grid', 'Scope 2', 0.295, 'kg CO2e/kWh', 'EEA 2024', 'EU', 2024, '2024-01-01'),
        ('APAC Grid', 'Scope 2', 0.408, 'kg CO2e/kWh', 'Regional Average', 'APAC', 2024, '2024-01-01'),
        ('Natural Gas', 'Scope 1', 56.1, 'kg CO2e/MMBtu', 'EPA GHG Inventory', 'Global', 2024, '2024-01-01'),
    ]
    
    cursor.executemany('INSERT INTO emission_factors VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?)', factors)
    print(f"✅ {len(factors)} factors")
    
    # Targets
    print("\n🎯 Setting targets...")
    
    baseline_scope12 = ANNUAL_EMISSIONS[2020]['scope1'] + ANNUAL_EMISSIONS[2020]['scope2_market']
    baseline_scope3 = ANNUAL_EMISSIONS[2020]['scope3']
    
    targets = [
        ('Scope 1 & 2', 2020, baseline_scope12, 2030, 50, baseline_scope12 * 0.5, True),
        ('Scope 3', 2020, baseline_scope3, 2030, 30, baseline_scope3 * 0.7, True),
    ]
    
    cursor.executemany('INSERT INTO emission_targets VALUES (NULL, ?, ?, ?, ?, ?, ?, ?)', targets)
    print("✅ SBTi targets set")
    
    # Scope 3 categories
    print("\n🔗 Loading Scope 3 categories...")
    
    scope3 = [
        (1, 'Purchased Goods & Services', 'Servers, equipment', 30.0),
        (2, 'Capital Goods', 'Data center construction', 40.0),
        (3, 'Fuel & Energy Related', 'Upstream electricity', 10.0),
        (4, 'Upstream Transportation', 'Logistics', 7.0),
        (6, 'Business Travel', 'Employee travel', 3.0),
        (7, 'Employee Commute', 'Commuting', 1.0),
        (11, 'Use of Sold Products', 'Cloud services', 8.0),
        (15, 'Investments', 'Portfolio', 1.0),
    ]
    
    cursor.executemany('INSERT INTO scope3_categories VALUES (?, ?, ?, ?)', scope3)
    print(f"✅ {len(scope3)} categories")
    
    # Generate time series
    print("\n📅 Generating monthly data (Jan 2020 - Oct 2025)...")
    
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2025, 10, 31)
    
    emissions_records = []
    operational_records = []
    
    np.random.seed(42)
    
    current_date = start_date
    while current_date <= end_date:
        # Get monthly allocation
        monthly = get_monthly_allocation(current_date)
        year = current_date.year
        
        for facility in facilities:
            facility_type = facility['type']
            facility_share = FACILITY_ALLOCATION[facility_type]
            region = facility['region']
            
            # Count facilities of this type
            type_count = sum(1 for f in facilities if f['type'] == facility_type)
            facility_fraction = facility_share / type_count
            
            # Variation
            variation = np.random.uniform(0.95, 1.05)
            
            # Emissions
            scope1 = monthly['scope1'] * facility_fraction * variation
            scope2_location = monthly['scope2_location'] * facility_fraction * variation
            scope2_market = monthly['scope2_market'] * facility_fraction * variation
            scope3 = monthly['scope3'] * facility_fraction * variation
            
            # Energy
            electricity = monthly['electricity'] * facility_fraction * variation
            fuel = monthly['fuel'] * facility_fraction * variation
            renewable_elec = electricity * monthly['renewable_pct']
            
            # PUE (only for data centers)
            if facility_type == 'Data Center' and facility['id'] in FACILITY_PUE_2024:
                # Use actual PUE, with slight improvement over time
                pue_2024 = FACILITY_PUE_2024[facility['id']]
                years_from_2024 = (current_date.year - 2024)
                pue = pue_2024 + (years_from_2024 * 0.002)  # Slight degradation before 2024, improvement after
            else:
                pue = None
            
            # CFE %
            cfe_pct = REGIONAL_CFE.get(region, 0.65)
            
            # Water (use actual 2024 data, back-calculate for earlier years)
            if facility_type == 'Data Center' and facility['id'] in FACILITY_WATER_2024:
                water_2024_annual = FACILITY_WATER_2024[facility['id']]
                # Scale based on year's total vs 2024 total
                year_factor = ANNUAL_WATER[year]['consumption'] / ANNUAL_WATER[2024]['consumption']
                water_consumption = (water_2024_annual / 12) * year_factor * variation
            else:
                water_consumption = monthly['water_consumption'] * facility_fraction * variation
            
            # Water withdrawal/discharge (typical ratio)
            water_withdrawal = water_consumption * 1.35
            water_discharge = water_withdrawal - water_consumption
            water_replenishment_pct = monthly['water_replenishment_pct']
            
            # Waste
            waste_total = monthly['waste_total'] * facility_fraction * variation
            waste_diverted = waste_total * monthly['waste_diversion_pct']
            
            # Costs
            energy_cost = electricity * ENERGY_COSTS.get(region, 70) / 1000  # $ thousands
            water_cost = water_consumption * WATER_COSTS.get(region, 8.5) / 1000  # $ thousands
            carbon_cost = (scope1 + scope2_market) * CARBON_PRICING.get(region, 0) / 1000  # $ thousands
            
            emissions_records.append((
                facility['id'],
                current_date.strftime('%Y-%m-%d'),
                scope1,
                scope2_location,
                scope2_market,
                scope3,
                electricity,
                monthly['renewable_pct'] * 100
            ))
            
            operational_records.append((
                facility['id'],
                current_date.strftime('%Y-%m-%d'),
                fuel,
                renewable_elec,
                cfe_pct,
                pue,
                water_withdrawal,
                water_discharge,
                water_consumption,
                water_replenishment_pct,
                waste_total,
                waste_diverted,
                monthly['waste_diversion_pct'],
                energy_cost,
                water_cost,
                carbon_cost
            ))
        
        # Next month
        current_date = current_date + timedelta(days=32)
        current_date = current_date.replace(day=1)
    
    # Insert data
    cursor.executemany('''
        INSERT INTO emissions_monthly 
        (facility_id, date, scope1_tonnes, scope2_location_tonnes, scope2_market_tonnes,
         scope3_tonnes, electricity_mwh, renewable_pct)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', emissions_records)
    
    cursor.executemany('''
        INSERT INTO facility_operational_metrics
        (facility_id, date, fuel_consumption_mwh, renewable_electricity_mwh, cfe_pct, pue,
         water_withdrawal_gallons, water_discharge_gallons, water_consumption_gallons, water_replenishment_pct,
         waste_generated_tons, waste_diverted_tons, waste_diversion_pct,
         energy_cost_usd, water_cost_usd, carbon_cost_usd)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', operational_records)
    
    print(f"✅ {len(emissions_records)} emission records")
    print(f"✅ {len(operational_records)} operational records")
    
    conn.commit()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    stats = cursor.execute('''
        SELECT 
            COUNT(DISTINCT facility_id) as facilities,
            COUNT(DISTINCT date) as months,
            ROUND(SUM(scope1_tonnes), 0) as s1,
            ROUND(SUM(scope2_market_tonnes), 0) as s2,
            ROUND(SUM(scope3_tonnes), 0) as s3,
            ROUND(SUM(scope1_tonnes + scope2_market_tonnes + scope3_tonnes), 0) as total
        FROM emissions_monthly
    ''').fetchone()
    
    print(f"\n📊 Database:")
    print(f"   Facilities: {stats[0]}")
    print(f"   Months: {stats[1]}")
    print(f"   Records: {len(emissions_records):,}")
    
    print(f"\n📈 Total Emissions:")
    print(f"   Scope 1: {stats[2]:,.0f} tonnes")
    print(f"   Scope 2: {stats[3]:,.0f} tonnes")
    print(f"   Scope 3: {stats[4]:,.0f} tonnes")
    print(f"   TOTAL: {stats[5]:,.0f} tonnes")
    
    # Verify against actual
    print(f"\n✅ Verification:")
    for year in [2020, 2021, 2022, 2023, 2024]:
        year_stats = cursor.execute('''
            SELECT ROUND(SUM(scope1_tonnes + scope2_market_tonnes + scope3_tonnes), 0)
            FROM emissions_monthly
            WHERE strftime('%Y', date) = ?
        ''', (str(year),)).fetchone()[0]
        
        actual = ANNUAL_EMISSIONS[year]['total']
        diff_pct = abs(year_stats - actual) / actual * 100
        print(f"   {year}: Generated={year_stats:,.0f} | Actual={actual:,.0f} | Diff={diff_pct:.1f}%")
    
    conn.close()
    
    print("\n✅ Database complete: sustainability_data.db")
    print("=" * 80)

if __name__ == "__main__":
    generate_sustainability_database()