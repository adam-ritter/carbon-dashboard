"""
Generate realistic sample sustainability data for the dashboard
Based on typical tech company emissions patterns
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sustainability_database():
    """Generate complete sustainability database with realistic data"""
    
    # Connect to database
    conn = sqlite3.connect('sustainability_data.db')
    cursor = conn.cursor()
    
    print("=" * 80)
    print("GENERATING CORPORATE SUSTAINABILITY DATABASE")
    print("=" * 80)
    
    # Create schema
    print("\n📊 Creating database schema...")
    
    cursor.executescript('''
    DROP TABLE IF EXISTS emissions_monthly;
    DROP TABLE IF EXISTS facilities;
    DROP TABLE IF EXISTS business_metrics;
    DROP TABLE IF EXISTS emission_factors;
    DROP TABLE IF EXISTS emission_targets;
    
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
    
    # Generate facilities
    print("\n🏭 Generating facility data...")
    
    facilities = [
        {'id': 'DC-VA-001', 'name': 'Virginia Data Center', 'region': 'US-East', 'type': 'Data Center'},
        {'id': 'DC-OR-001', 'name': 'Oregon Data Center', 'region': 'US-West', 'type': 'Data Center'},
        {'id': 'DC-IE-001', 'name': 'Dublin Data Center', 'region': 'EU', 'type': 'Data Center'},
        {'id': 'DC-SG-001', 'name': 'Singapore Data Center', 'region': 'APAC', 'type': 'Data Center'},
        {'id': 'OFF-WA-001', 'name': 'Seattle Campus', 'region': 'US-West', 'type': 'Office'},
        {'id': 'OFF-NY-001', 'name': 'New York Office', 'region': 'US-East', 'type': 'Office'},
        {'id': 'OFF-UK-001', 'name': 'London Office', 'region': 'EU', 'type': 'Office'},
        {'id': 'MFG-CN-001', 'name': 'Shenzhen Manufacturing', 'region': 'APAC', 'type': 'Manufacturing'},
    ]
    
    for facility in facilities:
        cursor.execute('''
        INSERT INTO facilities VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            facility['id'],
            facility['name'],
            facility['region'],
            facility['type'],
            '2020-01-01',
            3100 if facility['type'] == 'Data Center' else None
        ))
    
    print(f"✅ Generated {len(facilities)} facilities")
    
    # Generate emission factors
    print("\n📚 Loading emission factors...")
    
    emission_factors_data = [
        ('US Grid Average', 'Scope 2', 0.386, 'kg CO2e/kWh', 'EPA eGRID', 'United States', 2024, '2024-01-01'),
        ('California Grid', 'Scope 2', 0.203, 'kg CO2e/kWh', 'CARB', 'California', 2024, '2024-01-01'),
        ('Texas Grid (ERCOT)', 'Scope 2', 0.390, 'kg CO2e/kWh', 'EPA eGRID', 'Texas', 2024, '2024-01-01'),
        ('EU Grid Average', 'Scope 2', 0.295, 'kg CO2e/kWh', 'EEA', 'European Union', 2024, '2024-01-01'),
        ('Singapore Grid', 'Scope 2', 0.408, 'kg CO2e/kWh', 'EMA', 'Singapore', 2024, '2024-01-01'),
        ('Natural Gas Combustion', 'Scope 1', 56.1, 'kg CO2e/MMBtu', 'EPA', 'United States', 2024, '2024-01-01'),
        ('Diesel Combustion', 'Scope 1', 74.1, 'kg CO2e/MMBtu', 'EPA', 'United States', 2024, '2024-01-01'),
        ('RNG Production', 'Scope 3', 0.45, 'kg CO2e/kg RNG', 'EPA', 'United States', 2024, '2024-01-01'),
        ('Electricity T&D Losses', 'Scope 3', 0.031, 'kg CO2e/kWh', 'EPA', 'United States', 2024, '2024-01-01'),
        ('Business Travel - Air', 'Scope 3', 0.255, 'kg CO2e/passenger-km', 'DEFRA', 'Global', 2024, '2024-01-01'),
        ('Employee Commute - Car', 'Scope 3', 0.192, 'kg CO2e/km', 'EPA', 'United States', 2024, '2024-01-01'),
    ]
    
    cursor.executemany('''
    INSERT INTO emission_factors 
        (factor_name, scope, emission_factor, unit, source, geography, year, last_updated)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', emission_factors_data)
    
    print(f"✅ Loaded {len(emission_factors_data)} emission factors")
    
    # Generate targets
    print("\n🎯 Setting emission targets...")
    
    targets_data = [
        ('Scope 1 & 2', 2020, 3000000, 2030, 50, 1500000, True),
        ('Scope 3', 2020, 11000000, 2030, 30, 7700000, True),
    ]
    
    cursor.executemany('''
    INSERT INTO emission_targets
        (scope, baseline_year, baseline_emissions, target_year, reduction_percent, target_emissions, sbti_aligned)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', targets_data)
    
    print("✅ Targets set")
    
    # Generate Scope 3 categories
    print("\n🔗 Loading Scope 3 categories...")

    scope3_categories_data = [
        (1, 'Purchased Goods & Services', 'Servers, networking equipment, construction materials', 45.0),
        (2, 'Capital Goods', 'Data center construction, major equipment', 15.0),
        (3, 'Fuel & Energy Related', 'Upstream emissions from electricity generation', 8.0),
        (4, 'Upstream Transportation', 'Logistics and distribution', 3.0),
        (6, 'Business Travel', 'Employee air travel, hotels', 2.0),
        (7, 'Employee Commute', 'Daily commuting', 1.0),
        (11, 'Use of Sold Products', 'Customer use of cloud services', 25.0),
        (15, 'Investments', 'Equity investments', 1.0),
    ]

    cursor.executemany('''
    INSERT INTO scope3_categories
        (category_id, category_name, description, typical_pct_of_scope3)
    VALUES (?, ?, ?, ?)
    ''', scope3_categories_data)

    print(f"✅ Loaded {len(scope3_categories_data)} Scope 3 categories")

    # Generate time series data (36 months)
    print("\n📅 Generating 36 months of emissions data...")
    
    start_date = datetime(2021, 1, 1)
    months = 54
    
    emissions_records = []
    business_records = []
    
    for month_offset in range(months):
        current_date = start_date + timedelta(days=30 * month_offset)
        
        # Seasonal factors
        month_num = current_date.month
        summer_factor = 1.15 if month_num in [6, 7, 8] else 1.0
        winter_factor = 1.10 if month_num in [12, 1, 2] else 1.0
        
        # Growth trend
        growth_factor = 1 + (0.05 * month_offset / 12)  # 5% annual growth
        
        # Renewable energy ramp-up
        renewable_pct = min(0.85, 0.50 + (0.35 * month_offset / 36))
        
        for facility in facilities:
            facility_id = facility['id']
            facility_type = facility['type']
            region = facility['region']
            
            # Base emissions by facility type
            if facility_type == 'Data Center':
                scope1_base = 150  # Backup generators
                scope2_location_base = 8000  # Heavy electricity use
                scope3_base = 2500
                electricity_base = 20000  # MWh
                revenue_base = 50  # $M
                headcount_base = 200
                sqft_base = 500000
                servers_base = 10000
                
            elif facility_type == 'Office':
                scope1_base = 20
                scope2_location_base = 300
                scope3_base = 800
                electricity_base = 800
                revenue_base = 10
                headcount_base = 1000
                sqft_base = 200000
                servers_base = 100
                
            else:  # Manufacturing
                scope1_base = 500
                scope2_location_base = 1200
                scope3_base = 5000
                electricity_base = 3000
                revenue_base = 80
                headcount_base = 500
                sqft_base = 300000
                servers_base = 50
            
            # Regional electricity emission factors
            if region == 'US-East':
                grid_ef = 0.386
            elif region == 'US-West':
                grid_ef = 0.350
            elif region == 'EU':
                grid_ef = 0.295
            else:  # APAC
                grid_ef = 0.408
            
            # Apply factors and randomness
            scope1 = scope1_base * winter_factor * growth_factor * np.random.uniform(0.90, 1.10)
            
            electricity = electricity_base * summer_factor * growth_factor * np.random.uniform(0.95, 1.05)
            
            scope2_location = electricity * grid_ef
            scope2_market = scope2_location * (1 - renewable_pct)
            
            scope3 = scope3_base * growth_factor * np.random.uniform(0.90, 1.10)
            
            # Business metrics
            revenue = revenue_base * growth_factor * np.random.uniform(0.95, 1.05)
            headcount = int(headcount_base * growth_factor * np.random.uniform(0.98, 1.02))
            sqft = int(sqft_base * np.random.uniform(0.99, 1.01))
            servers = int(servers_base * growth_factor * np.random.uniform(0.95, 1.05))
            production = revenue * np.random.uniform(0.8, 1.2)  # Proxy
            
            # Store records
            emissions_records.append((
                facility_id,
                current_date.strftime('%Y-%m-%d'),
                scope1,
                scope2_location,
                scope2_market,
                scope3,
                electricity,
                renewable_pct * 100
            ))
            
            business_records.append((
                facility_id,
                current_date.strftime('%Y-%m-%d'),
                revenue,
                headcount,
                sqft,
                servers,
                production
            ))
    
    # Insert emissions data
    cursor.executemany('''
    INSERT INTO emissions_monthly
        (facility_id, date, scope1_tonnes, scope2_location_tonnes, scope2_market_tonnes,
         scope3_tonnes, electricity_mwh, renewable_pct)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', emissions_records)
    
    print(f"✅ Generated {len(emissions_records)} emission records")
    
    # Insert business metrics
    cursor.executemany('''
    INSERT INTO business_metrics
        (facility_id, date, revenue_millions, headcount, square_feet, server_count, production_volume)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', business_records)
    
    print(f"✅ Generated {len(business_records)} business metric records")
    
    # Commit and close
    conn.commit()
    
    # Generate summary statistics
    print("\n" + "=" * 80)
    print("DATABASE SUMMARY")
    print("=" * 80)
    
    summary_stats = cursor.execute('''
    SELECT 
        COUNT(DISTINCT facility_id) as facilities,
        COUNT(DISTINCT date) as months,
        COUNT(*) as total_records,
        ROUND(SUM(scope1_tonnes), 0) as total_scope1,
        ROUND(SUM(scope2_market_tonnes), 0) as total_scope2,
        ROUND(SUM(scope3_tonnes), 0) as total_scope3,
        ROUND(SUM(scope1_tonnes + scope2_market_tonnes + scope3_tonnes), 0) as total_emissions
    FROM emissions_monthly
    ''').fetchone()
    
    print(f"\n📊 Data Generated:")
    print(f"   Facilities:       {summary_stats[0]}")
    print(f"   Months:           {summary_stats[1]}")
    print(f"   Total Records:    {summary_stats[2]:,}")
    print(f"\n📈 Total Emissions (36 months):")
    print(f"   Scope 1:          {summary_stats[3]:,.0f} tonnes CO₂e")
    print(f"   Scope 2 (Market): {summary_stats[4]:,.0f} tonnes CO₂e")
    print(f"   Scope 3:          {summary_stats[5]:,.0f} tonnes CO₂e")
    print(f"   TOTAL:            {summary_stats[6]:,.0f} tonnes CO₂e")
    
    avg_monthly = cursor.execute('''
    SELECT 
        ROUND(AVG(scope1_tonnes + scope2_market_tonnes + scope3_tonnes), 0) as avg_monthly
    FROM emissions_monthly
    ''').fetchone()[0]
    
    print(f"\n💡 Average Monthly Emissions: {avg_monthly:,.0f} tonnes CO₂e")
    
    conn.close()
    
    print("\n✅ Database generation complete!")
    print(f"📁 Saved to: sustainability_data.db")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    generate_sustainability_database()