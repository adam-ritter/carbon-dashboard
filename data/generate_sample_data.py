"""
Generate sustainability data based on major tech company environmental reports

Data Source: Large tech company environmental reports (2020-2024 actual data)
Methodology: Monthly disaggregation of annual totals with realistic patterns

Annual Data (tCO2e):
2020: 8,737,400 total (Scope 1: 55,800 | Scope 2: 911,600 | Scope 3: 7,770,000)
2021: 10,520,600 total (Scope 1: 64,100 | Scope 2: 1,823,500 | Scope 3: 8,633,000)
2022: 11,900,300 total (Scope 1: 91,200 | Scope 2: 2,492,100 | Scope 3: 9,317,000)
2023: 14,296,800 total (Scope 1: 79,400 | Scope 2: 3,423,400 | Scope 3: 10,794,000)
2024: 15,185,200 total (Scope 1: 73,100 | Scope 2: 3,059,100 | Scope 3: 12,053,000)

Renewable Energy: 100% electricity matched annually
Carbon-Free Energy (CFE): 66% hourly match (2024)
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Actual annual data from environmental reports
ANNUAL_DATA = {
    2020: {
        'scope1': 55_800,
        'scope2_location': 5_865_100,
        'scope2_market': 911_600,
        'scope3': 7_770_000,
        'total': 8_737_400,
        'renewable_pct': 1.00,  # 100% renewable match
        'cfe_pct': 0.67  # 67% carbon-free energy (hourly)
    },
    2021: {
        'scope1': 64_100,
        'scope2_location': 6_576_200,
        'scope2_market': 1_823_500,
        'scope3': 8_633_000,
        'total': 10_520_600,
        'renewable_pct': 1.00,
        'cfe_pct': 0.65
    },
    2022: {
        'scope1': 91_200,
        'scope2_location': 8_045_400,
        'scope2_market': 2_492_100,
        'scope3': 9_317_000,
        'total': 11_900_300,
        'renewable_pct': 1.00,
        'cfe_pct': 0.64
    },
    2023: {
        'scope1': 79_400,
        'scope2_location': 9_252_900,
        'scope2_market': 3_423_400,
        'scope3': 10_794_000,
        'total': 14_296_800,
        'renewable_pct': 1.00,
        'cfe_pct': 0.64
    },
    2024: {
        'scope1': 73_100,
        'scope2_location': 11_283_200,
        'scope2_market': 3_059_100,
        'scope3': 12_053_000,
        'total': 15_185_200,
        'renewable_pct': 1.00,
        'cfe_pct': 0.66
    },
    2025: {  # Projection based on trends
        'scope1': 70_000,
        'scope2_location': 12_500_000,
        'scope2_market': 2_800_000,
        'scope3': 13_000_000,
        'total': 15_870_000,
        'renewable_pct': 1.00,
        'cfe_pct': 0.68
    }
}

# Regional grid emission factors (kg CO2e/kWh)
GRID_FACTORS = {
    'US-East': 0.386,
    'US-West': 0.203,
    'US-Central': 0.390,
    'EU': 0.295,
    'APAC': 0.408
}

# Facility type allocation (based on typical tech company breakdown)
FACILITY_ALLOCATION = {
    'Data Center': 0.85,  # 85% of operational emissions
    'Office': 0.10,       # 10% of operational emissions
    'Manufacturing': 0.05  # 5% of operational emissions
}

def get_monthly_data(date):
    """
    Allocate annual totals to months with seasonal patterns
    """
    year = date.year
    month = date.month
    
    # Get annual data
    if year in ANNUAL_DATA:
        annual = ANNUAL_DATA[year]
    else:
        annual = ANNUAL_DATA[2024]  # Default to latest
    
    # Monthly base: annual / 12
    monthly_scope1 = annual['scope1'] / 12
    monthly_scope2_location = annual['scope2_location'] / 12
    monthly_scope2_market = annual['scope2_market'] / 12
    monthly_scope3 = annual['scope3'] / 12
    
    # Seasonal factors (data centers have cooling/heating cycles)
    if month in [6, 7, 8]:  # Summer - higher cooling demand
        seasonal_factor = 1.08
    elif month in [12, 1, 2]:  # Winter - higher heating demand
        seasonal_factor = 1.05
    else:
        seasonal_factor = 1.0
    
    return {
        'scope1': monthly_scope1 * seasonal_factor,
        'scope2_location': monthly_scope2_location * seasonal_factor,
        'scope2_market': monthly_scope2_market * seasonal_factor,
        'scope3': monthly_scope3,  # Scope 3 less seasonal
        'renewable_pct': annual['renewable_pct'],
        'cfe_pct': annual['cfe_pct']
    }

def generate_sustainability_database():
    """Generate database based on actual environmental report data"""
    
    print("=" * 80)
    print("SUSTAINABILITY DATABASE GENERATOR")
    print("Based on Tech Industry Environmental Reports (2020-2024)")
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
    
    CREATE TABLE business_metrics (
        metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
        facility_id TEXT,
        date DATE,
        revenue_millions REAL,
        headcount INTEGER,
        square_feet INTEGER,
        server_count INTEGER,
        production_volume REAL,
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
        # Data Centers (85% of emissions)
        {'id': 'DC-VA-001', 'name': 'Virginia Data Center', 'region': 'US-East', 'type': 'Data Center'},
        {'id': 'DC-VA-002', 'name': 'Virginia Data Center 2', 'region': 'US-East', 'type': 'Data Center'},
        {'id': 'DC-OR-001', 'name': 'Oregon Data Center', 'region': 'US-West', 'type': 'Data Center'},
        {'id': 'DC-CA-001', 'name': 'California Data Center', 'region': 'US-West', 'type': 'Data Center'},
        {'id': 'DC-TX-001', 'name': 'Texas Data Center', 'region': 'US-Central', 'type': 'Data Center'},
        {'id': 'DC-IE-001', 'name': 'Dublin Data Center', 'region': 'EU', 'type': 'Data Center'},
        {'id': 'DC-NL-001', 'name': 'Amsterdam Data Center', 'region': 'EU', 'type': 'Data Center'},
        {'id': 'DC-SG-001', 'name': 'Singapore Data Center', 'region': 'APAC', 'type': 'Data Center'},
        {'id': 'DC-JP-001', 'name': 'Tokyo Data Center', 'region': 'APAC', 'type': 'Data Center'},
        # Offices (10% of emissions)
        {'id': 'OFF-WA-001', 'name': 'Seattle Campus', 'region': 'US-West', 'type': 'Office'},
        {'id': 'OFF-CA-001', 'name': 'San Francisco Office', 'region': 'US-West', 'type': 'Office'},
        {'id': 'OFF-NY-001', 'name': 'New York Office', 'region': 'US-East', 'type': 'Office'},
        {'id': 'OFF-TX-001', 'name': 'Austin Office', 'region': 'US-Central', 'type': 'Office'},
        {'id': 'OFF-UK-001', 'name': 'London Office', 'region': 'EU', 'type': 'Office'},
        {'id': 'OFF-DE-001', 'name': 'Berlin Office', 'region': 'EU', 'type': 'Office'},
        {'id': 'OFF-SG-001', 'name': 'Singapore Office', 'region': 'APAC', 'type': 'Office'},
        # Manufacturing (5% of emissions)
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
        ('Diesel', 'Scope 1', 74.1, 'kg CO2e/MMBtu', 'EPA GHG Inventory', 'Global', 2024, '2024-01-01'),
    ]
    
    cursor.executemany('INSERT INTO emission_factors VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?)', factors)
    print(f"✅ {len(factors)} factors")
    
    # Targets (based on 2020 baseline, 2030 goals)
    print("\n🎯 Setting targets...")
    
    # Calculate baseline from actual 2020 data
    baseline_scope12 = ANNUAL_DATA[2020]['scope1'] + ANNUAL_DATA[2020]['scope2_market']
    baseline_scope3 = ANNUAL_DATA[2020]['scope3']
    
    targets = [
        ('Scope 1 & 2', 2020, baseline_scope12, 2030, 50, baseline_scope12 * 0.5, True),
        ('Scope 3', 2020, baseline_scope3, 2030, 30, baseline_scope3 * 0.7, True),
    ]
    
    cursor.executemany('INSERT INTO emission_targets VALUES (NULL, ?, ?, ?, ?, ?, ?, ?)', targets)
    print("✅ SBTi-aligned targets (50% Scope 1&2, 30% Scope 3 by 2030)")
    
    # Scope 3 categories (based on actual breakdown)
    print("\n🔗 Loading Scope 3 categories...")
    
    scope3 = [
        (1, 'Purchased Goods & Services', 'Servers, equipment, materials', 30.0),
        (2, 'Capital Goods', 'Data center construction', 40.0),  # Includes Cat 11 (Use of sold products)
        (3, 'Fuel & Energy Related', 'Upstream electricity', 10.0),
        (4, 'Upstream Transportation', 'Logistics', 7.0),
        (6, 'Business Travel', 'Employee travel', 3.0),
        (7, 'Employee Commute', 'Daily commuting', 1.0),
        (11, 'Use of Sold Products', 'Cloud services use', 8.0),
        (15, 'Investments', 'Portfolio companies', 1.0),
    ]
    
    cursor.executemany('INSERT INTO scope3_categories VALUES (?, ?, ?, ?)', scope3)
    print(f"✅ {len(scope3)} categories")
    
    # Generate time series
    print("\n📅 Generating monthly data (Jan 2020 - Oct 2025)...")
    print(f"    Total emissions growth: {ANNUAL_DATA[2020]['total']:,.0f} → {ANNUAL_DATA[2024]['total']:,.0f} tonnes")
    print(f"    Renewable energy: {ANNUAL_DATA[2020]['renewable_pct']*100:.0f}% maintained throughout")
    print(f"    Carbon-free energy (hourly): {ANNUAL_DATA[2020]['cfe_pct']*100:.0f}% → {ANNUAL_DATA[2024]['cfe_pct']*100:.0f}%")
    
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2025, 10, 31)
    
    emissions_records = []
    business_records = []
    
    np.random.seed(42)
    
    current_date = start_date
    while current_date <= end_date:
        # Get monthly allocation from annual data
        monthly_data = get_monthly_data(current_date)
        
        # Allocate to facilities
        for facility in facilities:
            facility_type = facility['type']
            facility_share = FACILITY_ALLOCATION[facility_type]
            
            # Count facilities of this type
            type_count = sum(1 for f in facilities if f['type'] == facility_type)
            
            # This facility's share (equal distribution within type)
            facility_fraction = facility_share / type_count
            
            # Add facility-level variation (±5%)
            variation = np.random.uniform(0.95, 1.05)
            
            # Calculate emissions
            scope1 = monthly_data['scope1'] * facility_fraction * variation
            scope2_location = monthly_data['scope2_location'] * facility_fraction * variation
            scope2_market = monthly_data['scope2_market'] * facility_fraction * variation
            scope3 = monthly_data['scope3'] * facility_fraction * variation
            
            # Calculate electricity from Scope 2 location-based
            grid_ef = GRID_FACTORS[facility['region']]
            electricity_mwh = scope2_location / grid_ef if grid_ef > 0 else 0
            
            # Business metrics (simple allocation based on facility type)
            if facility_type == 'Data Center':
                revenue = 50 * variation
                headcount = 200
                sqft = 500000
                servers = 10000
            elif facility_type == 'Office':
                revenue = 10 * variation
                headcount = 800
                sqft = 150000
                servers = 100
            else:  # Manufacturing
                revenue = 30 * variation
                headcount = 400
                sqft = 250000
                servers = 50
            
            emissions_records.append((
                facility['id'],
                current_date.strftime('%Y-%m-%d'),
                scope1,
                scope2_location,
                scope2_market,
                scope3,
                electricity_mwh,
                monthly_data['renewable_pct'] * 100
            ))
            
            business_records.append((
                facility['id'],
                current_date.strftime('%Y-%m-%d'),
                revenue,
                headcount,
                sqft,
                servers,
                revenue * 1.2  # Production volume proxy
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
        INSERT INTO business_metrics
        (facility_id, date, revenue_millions, headcount, square_feet, server_count, production_volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', business_records)
    
    print(f"✅ {len(emissions_records)} emission records")
    print(f"✅ {len(business_records)} business records")
    
    conn.commit()
    
    # Summary and verification
    print("\n" + "=" * 80)
    print("SUMMARY & VERIFICATION")
    print("=" * 80)
    
    stats = cursor.execute('''
        SELECT 
            COUNT(DISTINCT facility_id) as facilities,
            COUNT(DISTINCT date) as months,
            ROUND(SUM(scope1_tonnes), 0) as total_s1,
            ROUND(SUM(scope2_market_tonnes), 0) as total_s2,
            ROUND(SUM(scope3_tonnes), 0) as total_s3,
            ROUND(SUM(scope1_tonnes + scope2_market_tonnes + scope3_tonnes), 0) as total
        FROM emissions_monthly
    ''').fetchone()
    
    print(f"\n📊 Database Generated:")
    print(f"   Facilities: {stats[0]}")
    print(f"   Months: {stats[1]} (Jan 2020 - Oct 2025)")
    print(f"   Total Records: {len(emissions_records):,}")
    
    print(f"\n📈 Total Emissions (All periods):")
    print(f"   Scope 1: {stats[2]:,.0f} tonnes")
    print(f"   Scope 2 (Market): {stats[3]:,.0f} tonnes")
    print(f"   Scope 3: {stats[4]:,.0f} tonnes")
    print(f"   TOTAL: {stats[5]:,.0f} tonnes")
    
    # Verify against actual annual data
    print(f"\n✅ Verification Against Published Data:")
    
    for year in [2020, 2021, 2022, 2023, 2024]:
        year_stats = cursor.execute('''
            SELECT 
                ROUND(SUM(scope1_tonnes), 0) as s1,
                ROUND(SUM(scope2_market_tonnes), 0) as s2,
                ROUND(SUM(scope3_tonnes), 0) as s3,
                ROUND(SUM(scope1_tonnes + scope2_market_tonnes + scope3_tonnes), 0) as total
            FROM emissions_monthly
            WHERE strftime('%Y', date) = ?
        ''', (str(year),)).fetchone()
        
        actual = ANNUAL_DATA[year]
        diff_pct = abs(year_stats[3] - actual['total']) / actual['total'] * 100
        
        print(f"   {year}: Generated={year_stats[3]:,.0f} | Actual={actual['total']:,.0f} | Diff={diff_pct:.1f}%")
    
    conn.close()
    
    print("\n✅ Database complete: sustainability_data.db")
    print("=" * 80)

if __name__ == "__main__":
    generate_sustainability_database()