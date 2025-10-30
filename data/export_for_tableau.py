"""
Export cleaned sustainability data for Tableau
"""

import sqlite3
import pandas as pd
import os

def export_data_for_tableau():
    """Export all tables to CSV for Tableau import"""
    
    db_path = 'sustainability_data.db'
    
    conn = sqlite3.connect(db_path)
    
    # Create exports directory
    export_dir = 'tableau_exports'
    os.makedirs(export_dir, exist_ok=True)
    
    print(f"\n📁 Exporting to: {export_dir}/")
    print("=" * 60)
    
    # 1. Main emissions data with all dimensions
    print("\n1. Exporting main emissions dataset...")
    emissions_query = """
    SELECT 
        e.date,
        e.facility_id,
        f.facility_name,
        f.region,
        f.facility_type,
        e.scope1_tonnes,
        e.scope2_location_tonnes,
        e.scope2_market_tonnes,
        e.scope3_tonnes,
        (e.scope1_tonnes + e.scope2_market_tonnes + e.scope3_tonnes) as total_emissions,
        e.electricity_mwh,
        e.renewable_pct,
        -- Add year/month for easy filtering
        strftime('%Y', e.date) as year,
        strftime('%m', e.date) as month,
        strftime('%Y-%m', e.date) as year_month,
        -- Add business metrics
        b.revenue_millions,
        b.headcount,
        b.square_feet,
        b.server_count,
        -- Add intensity metrics
        ROUND((e.scope1_tonnes + e.scope2_market_tonnes + e.scope3_tonnes) / b.revenue_millions, 2) as intensity_per_revenue,
        ROUND((e.scope1_tonnes + e.scope2_market_tonnes + e.scope3_tonnes) / b.headcount, 2) as intensity_per_employee
    FROM emissions_monthly e
    JOIN facilities f ON e.facility_id = f.facility_id
    LEFT JOIN business_metrics b ON e.facility_id = b.facility_id AND e.date = b.date
    ORDER BY e.date, e.facility_id
    """
    
    df_emissions = pd.read_sql_query(emissions_query, conn)
    df_emissions.to_csv(f'{export_dir}/emissions_data.csv', index=False)
    print(f"   ✅ emissions_data.csv ({len(df_emissions):,} rows)")
    
    # 2. Facilities reference
    print("\n2. Exporting facilities reference...")
    facilities_query = """
    SELECT 
        facility_id,
        facility_name,
        region,
        facility_type,
        operational_start_date
    FROM facilities
    ORDER BY facility_name
    """
    
    df_facilities = pd.read_sql_query(facilities_query, conn)
    df_facilities.to_csv(f'{export_dir}/facilities.csv', index=False)
    print(f"   ✅ facilities.csv ({len(df_facilities):,} rows)")
    
    # 3. Emission targets
    print("\n3. Exporting emission targets...")
    targets_query = """
    SELECT 
        scope,
        baseline_year,
        baseline_emissions,
        target_year,
        reduction_percent,
        target_emissions,
        sbti_aligned
    FROM emission_targets
    """
    
    df_targets = pd.read_sql_query(targets_query, conn)
    df_targets.to_csv(f'{export_dir}/emission_targets.csv', index=False)
    print(f"   ✅ emission_targets.csv ({len(df_targets):,} rows)")
    
    # 4. Scope 3 categories
    print("\n4. Exporting Scope 3 categories...")
    scope3_query = """
    SELECT 
        category_id,
        category_name,
        description,
        typical_pct_of_scope3
    FROM scope3_categories
    """
    
    df_scope3 = pd.read_sql_query(scope3_query, conn)
    df_scope3.to_csv(f'{export_dir}/scope3_categories.csv', index=False)
    print(f"   ✅ scope3_categories.csv ({len(df_scope3):,} rows)")
    
    # 5. Monthly aggregated summary
    print("\n5. Exporting monthly summary...")
    monthly_query = """
    SELECT 
        strftime('%Y-%m', date) as year_month,
        strftime('%Y', date) as year,
        strftime('%m', date) as month,
        COUNT(DISTINCT facility_id) as facilities_reporting,
        SUM(scope1_tonnes) as total_scope1,
        SUM(scope2_market_tonnes) as total_scope2,
        SUM(scope3_tonnes) as total_scope3,
        SUM(scope1_tonnes + scope2_market_tonnes + scope3_tonnes) as total_emissions,
        SUM(electricity_mwh) as total_electricity,
        AVG(renewable_pct) as avg_renewable_pct
    FROM emissions_monthly
    GROUP BY strftime('%Y-%m', date)
    ORDER BY year_month
    """
    
    df_monthly = pd.read_sql_query(monthly_query, conn)
    df_monthly.to_csv(f'{export_dir}/monthly_summary.csv', index=False)
    print(f"   ✅ monthly_summary.csv ({len(df_monthly):,} rows)")
    
    # 6. Regional summary
    print("\n6. Exporting regional summary...")
    regional_query = """
    SELECT 
        f.region,
        strftime('%Y-%m', e.date) as year_month,
        COUNT(DISTINCT e.facility_id) as facilities,
        SUM(e.scope1_tonnes) as scope1,
        SUM(e.scope2_market_tonnes) as scope2,
        SUM(e.scope3_tonnes) as scope3,
        SUM(e.scope1_tonnes + e.scope2_market_tonnes + e.scope3_tonnes) as total_emissions
    FROM emissions_monthly e
    JOIN facilities f ON e.facility_id = f.facility_id
    GROUP BY f.region, strftime('%Y-%m', e.date)
    ORDER BY year_month, f.region
    """
    
    df_regional = pd.read_sql_query(regional_query, conn)
    df_regional.to_csv(f'{export_dir}/regional_summary.csv', index=False)
    print(f"   ✅ regional_summary.csv ({len(df_regional):,} rows)")
    
    # 7. Facility type summary
    print("\n7. Exporting facility type summary...")
    type_query = """
    SELECT 
        f.facility_type,
        strftime('%Y-%m', e.date) as year_month,
        COUNT(DISTINCT e.facility_id) as facilities,
        SUM(e.scope1_tonnes) as scope1,
        SUM(e.scope2_market_tonnes) as scope2,
        SUM(e.scope3_tonnes) as scope3,
        SUM(e.scope1_tonnes + e.scope2_market_tonnes + e.scope3_tonnes) as total_emissions
    FROM emissions_monthly e
    JOIN facilities f ON e.facility_id = f.facility_id
    GROUP BY f.facility_type, strftime('%Y-%m', e.date)
    ORDER BY year_month, f.facility_type
    """
    
    df_type = pd.read_sql_query(type_query, conn)
    df_type.to_csv(f'{export_dir}/facility_type_summary.csv', index=False)
    print(f"   ✅ facility_type_summary.csv ({len(df_type):,} rows)")
    
    conn.close()
    
    print("\n" + "=" * 60)
    print("✅ Export complete!")
    print(f"\n📊 Files ready for Tableau import in: {export_dir}/")
    print("\nNext steps:")
    print("1. Open Tableau Desktop")
    print("2. Connect to Data → Text file")
    print("3. Navigate to tableau_exports/ folder")
    print("4. Import emissions_data.csv as main data source")
    print("5. Add other CSVs as supplementary data sources")
    print("6. Create relationships/joins as needed")
    
    return export_dir

if __name__ == "__main__":
    export_data_for_tableau()