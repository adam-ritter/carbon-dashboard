import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import os
sys.path.append('..')
from utils.data_loader import load_emissions_data, load_facilities
import sqlite3

st.set_page_config(page_title="Data Quality", page_icon="💾", layout="wide")

st.markdown('<style>.main-header {font-size: 2.5rem; font-weight: 700; color: #9467bd;}</style>', unsafe_allow_html=True)
st.markdown('<p class="main-header">💾 Data Quality Assurance</p>', unsafe_allow_html=True)

st.markdown("""
## Automated Data Quality Checks

Comprehensive validation for sustainability reporting accuracy and completeness.

**Quality Dimensions:**
- Completeness (missing values)
- Accuracy (validation rules)
- Consistency (logical checks)
- Timeliness (data freshness)
- Validity (range checks)
""")

# Check if cleaned database exists
cleaned_db_path = '../data/sustainability_data_clean.db'
cleaned_db_exists = os.path.exists(cleaned_db_path)

if cleaned_db_exists:
    st.success("✅ **Cleaned database exists** - All analytics pages are using cleaned data")
    use_cleaned = True
else:
    st.warning("⚠️ **Using raw data** - Data contains quality issues. Generate cleaned database below.")
    use_cleaned = False

st.markdown("---")

# Connect to appropriate database
db_path = cleaned_db_path if use_cleaned else '../data/sustainability_data.db'
conn = sqlite3.connect(db_path)

@st.cache_data
def run_quality_checks(_conn):
    results = {}
    
    # Completeness check
    completeness_df = pd.read_sql_query("""
    WITH data_quality AS (
        SELECT 
            COUNT(*) as total_records,
            SUM(CASE WHEN scope1_tonnes IS NULL THEN 1 ELSE 0 END) as missing_scope1,
            SUM(CASE WHEN scope2_market_tonnes IS NULL THEN 1 ELSE 0 END) as missing_scope2,
            SUM(CASE WHEN scope3_tonnes IS NULL THEN 1 ELSE 0 END) as missing_scope3,
            SUM(CASE WHEN renewable_pct IS NULL THEN 1 ELSE 0 END) as missing_renewable
        FROM emissions_monthly
    )
    SELECT 
        'Total Records' as check_type, total_records as count, 100.0 as pct_pass
    FROM data_quality
    UNION ALL
    SELECT 'Complete Scope 1', total_records - missing_scope1, 
           ROUND((total_records - missing_scope1) * 100.0 / total_records, 2)
    FROM data_quality
    UNION ALL
    SELECT 'Complete Scope 2', total_records - missing_scope2,
           ROUND((total_records - missing_scope2) * 100.0 / total_records, 2)
    FROM data_quality
    UNION ALL
    SELECT 'Complete Scope 3', total_records - missing_scope3,
           ROUND((total_records - missing_scope3) * 100.0 / total_records, 2)
    FROM data_quality
    UNION ALL
    SELECT 'Complete Renewable %', total_records - missing_renewable,
           ROUND((total_records - missing_renewable) * 100.0 / total_records, 2)
    FROM data_quality
    """, _conn)
    results['completeness'] = completeness_df
    
    # Consistency checks
    consistency_df = pd.read_sql_query("""
    SELECT 
        COUNT(*) as total_records,
        SUM(CASE WHEN scope2_market_tonnes > scope2_location_tonnes THEN 1 ELSE 0 END) as inconsistent_scope2,
        SUM(CASE WHEN scope1_tonnes < 0 OR scope2_market_tonnes < 0 OR scope3_tonnes < 0 THEN 1 ELSE 0 END) as negative_values,
        SUM(CASE WHEN renewable_pct < 0 OR renewable_pct > 100 THEN 1 ELSE 0 END) as invalid_renewable,
        SUM(CASE WHEN electricity_mwh <= 0 THEN 1 ELSE 0 END) as zero_electricity
    FROM emissions_monthly
    """, _conn)
    results['consistency'] = consistency_df
    
    # Range checks
    range_df = pd.read_sql_query("""
    SELECT 
        facility_id,
        date,
        scope1_tonnes,
        scope2_market_tonnes,
        scope3_tonnes,
        renewable_pct,
        CASE 
            WHEN scope1_tonnes > (SELECT AVG(scope1_tonnes) * 3 FROM emissions_monthly WHERE scope1_tonnes > 0) THEN 'High Scope 1'
            WHEN scope2_market_tonnes > (SELECT AVG(scope2_market_tonnes) * 3 FROM emissions_monthly WHERE scope2_market_tonnes > 0) THEN 'High Scope 2'
            WHEN scope3_tonnes > (SELECT AVG(scope3_tonnes) * 3 FROM emissions_monthly WHERE scope3_tonnes > 0) THEN 'High Scope 3'
            ELSE 'Normal'
        END as flag
    FROM emissions_monthly
    WHERE scope1_tonnes > (SELECT AVG(scope1_tonnes) * 3 FROM emissions_monthly WHERE scope1_tonnes > 0)
       OR scope2_market_tonnes > (SELECT AVG(scope2_market_tonnes) * 3 FROM emissions_monthly WHERE scope2_market_tonnes > 0)
       OR scope3_tonnes > (SELECT AVG(scope3_tonnes) * 3 FROM emissions_monthly WHERE scope3_tonnes > 0)
    """, _conn)
    results['outliers'] = range_df
    
    # Timeliness check
    timeliness_df = pd.read_sql_query("""
    SELECT 
        MAX(date) as latest_date,
        julianday('now') - julianday(MAX(date)) as days_since_last_update,
        COUNT(DISTINCT facility_id) as facilities_reporting
    FROM emissions_monthly
    """, _conn)
    results['timeliness'] = timeliness_df
    
    # Monthly continuity check
    continuity_df = pd.read_sql_query("""
    WITH expected_months AS (
        SELECT DISTINCT 
            facility_id,
            date
        FROM emissions_monthly
    ),
    facility_months AS (
        SELECT 
            f.facility_id,
            f.facility_name,
            COUNT(DISTINCT strftime('%Y-%m', e.date)) as months_reported,
            (SELECT COUNT(DISTINCT strftime('%Y-%m', date)) FROM emissions_monthly) as expected_months
        FROM facilities f
        LEFT JOIN emissions_monthly e ON f.facility_id = e.facility_id
        GROUP BY f.facility_id
    )
    SELECT 
        facility_id,
        facility_name,
        months_reported,
        expected_months,
        ROUND(months_reported * 100.0 / expected_months, 1) as completeness_pct
    FROM facility_months
    WHERE months_reported < expected_months
    ORDER BY completeness_pct ASC
    """, _conn)
    results['continuity'] = continuity_df
    
    return results

try:
    st.markdown("### 🔍 Running Quality Checks...")
    with st.spinner("Analyzing data quality..."):
        results = run_quality_checks(conn)
    
    st.success("✅ Quality checks complete!")
    
    # Overall score
    completeness_score = results['completeness'][results['completeness']['check_type'] != 'Total Records']['pct_pass'].mean()
    consistency_df = results['consistency']
    total_records = consistency_df['total_records'].iloc[0]
    consistency_issues = (
        consistency_df['inconsistent_scope2'].iloc[0] +
        consistency_df['negative_values'].iloc[0] +
        consistency_df['invalid_renewable'].iloc[0] +
        consistency_df['zero_electricity'].iloc[0]
    )
    consistency_score = ((total_records - consistency_issues) / total_records) * 100
    
    outlier_count = len(results['outliers'])
    outlier_score = max(0, 100 - (outlier_count / total_records * 100))
    
    overall_score = (completeness_score + consistency_score + outlier_score) / 3
    
    st.markdown("### 🎯 Overall Data Quality Score")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        color = "normal" if overall_score >= 95 else "inverse"
        st.metric("Overall Score", f"{overall_score:.1f}%", 
                 "Excellent" if overall_score >= 95 else "Needs attention",
                 delta_color=color)
    
    with col2:
        st.metric("Completeness", f"{completeness_score:.1f}%",
                 help="Percentage of fields with non-null values")
    
    with col3:
        st.metric("Consistency", f"{consistency_score:.1f}%",
                 help="Records passing logical validation rules")
    
    with col4:
        st.metric("Outliers Detected", outlier_count,
                 help="Records flagged as potential anomalies")
    
    # Quality gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=overall_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Data Quality Score ({'Cleaned' if use_cleaned else 'Raw'})"},
        delta={'reference': 95, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 70], 'color': "lightgray"},
                {'range': [70, 85], 'color': "lightyellow"},
                {'range': [85, 95], 'color': "lightgreen"},
                {'range': [95, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 95
            }
        }
    ))
    fig_gauge.update_layout(height=300)
    st.plotly_chart(fig_gauge, width='stretch')
    
    st.markdown("---")
    
    # Completeness details
    st.markdown("### 📊 Completeness Analysis")
    st.markdown("Percentage of records with complete data for each field.")
    
    completeness_display = results['completeness'].copy()
    
    fig_complete = go.Figure(go.Bar(
        x=completeness_display['pct_pass'],
        y=completeness_display['check_type'],
        orientation='h',
        marker=dict(
            color=completeness_display['pct_pass'],
            colorscale='RdYlGn',
            cmin=0,
            cmax=100,
            showscale=True,
            colorbar=dict(title="% Complete")
        ),
        text=completeness_display['pct_pass'].apply(lambda x: f"{x:.1f}%"),
        textposition='outside'
    ))
    
    fig_complete.update_layout(
        title='Data Completeness by Field',
        xaxis_title='Completeness (%)',
        xaxis_range=[0, 105],
        height=400,
        template='plotly_white',
        yaxis={'categoryorder': 'total ascending'}
    )
    
    st.plotly_chart(fig_complete, width='stretch')
    st.dataframe(completeness_display, width='stretch')
    
    st.markdown("---")
    
    # Consistency checks
    st.markdown("### ✅ Consistency Validation")
    st.markdown("Logical validation rules applied to detect data quality issues.")
    
    consistency_summary = pd.DataFrame({
        'Check': [
            'Scope 2 Market ≤ Location',
            'No Negative Values',
            'Valid Renewable % (0-100)',
            'Non-Zero Electricity'
        ],
        'Issues Found': [
            results['consistency']['inconsistent_scope2'].iloc[0],
            results['consistency']['negative_values'].iloc[0],
            results['consistency']['invalid_renewable'].iloc[0],
            results['consistency']['zero_electricity'].iloc[0]
        ],
        'Total Records': [total_records] * 4
    })
    
    consistency_summary['Pass Rate %'] = (
        (consistency_summary['Total Records'] - consistency_summary['Issues Found']) / 
        consistency_summary['Total Records'] * 100
    )
    
    fig_consistency = go.Figure(go.Bar(
        x=consistency_summary['Pass Rate %'],
        y=consistency_summary['Check'],
        orientation='h',
        marker=dict(
            color=consistency_summary['Pass Rate %'],
            colorscale='RdYlGn',
            cmin=95,
            cmax=100
        ),
        text=consistency_summary['Pass Rate %'].apply(lambda x: f"{x:.2f}%"),
        textposition='outside'
    ))
    
    fig_consistency.update_layout(
        title='Consistency Check Pass Rates',
        xaxis_title='Pass Rate (%)',
        xaxis_range=[95, 105],
        height=350,
        template='plotly_white',
        yaxis={'categoryorder': 'total ascending'}
    )
    
    st.plotly_chart(fig_consistency, width='stretch')
    st.dataframe(consistency_summary, width='stretch')
    
    if consistency_issues > 0:
        st.warning(f"⚠️ Found {consistency_issues} consistency issues")
    else:
        st.success("✅ All records pass consistency checks!")
    
    st.markdown("---")
    
    # Outlier detection
    st.markdown("### 🚨 Outlier Detection")
    st.markdown("Records flagged as statistical outliers (>3x average).")
    
    if len(results['outliers']) > 0:
        st.warning(f"⚠️ {len(results['outliers'])} outliers detected")
        
        outliers_display = results['outliers'].copy()
        outliers_display['date'] = pd.to_datetime(outliers_display['date']).dt.strftime('%Y-%m')
        
        fig_outliers = px.scatter(
            outliers_display,
            x='date',
            y='scope1_tonnes',
            color='flag',
            size='scope3_tonnes',
            hover_data=['facility_id', 'scope2_market_tonnes'],
            title='Outlier Records by Date'
        )
        fig_outliers.update_layout(height=400, template='plotly_white')
        st.plotly_chart(fig_outliers, width='stretch')
        
        st.dataframe(outliers_display.head(20), width='stretch')
    else:
        st.success("✅ No outliers detected")
    
    st.markdown("---")
    
    # Timeliness
    st.markdown("### ⏰ Data Timeliness")
    
    timeliness = results['timeliness'].iloc[0]
    latest_date = pd.to_datetime(timeliness['latest_date'])
    days_old = int(timeliness['days_since_last_update'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Latest Data Date", latest_date.strftime('%Y-%m-%d'), f"{days_old} days ago")
    
    with col2:
        st.metric("Facilities Reporting", int(timeliness['facilities_reporting']))
    
    if days_old > 60:
        st.error(f"⚠️ Data is {days_old} days old")
    elif days_old > 30:
        st.warning(f"⚡ Data is {days_old} days old")
    else:
        st.success(f"✅ Data is current ({days_old} days old)")
    
    # DATA CLEANING SECTION
    st.markdown("---")
    st.markdown("### 🧹 Data Cleaning & Database Generation")
    
    if not use_cleaned and consistency_issues > 0:
        st.markdown("""
        **Detected Issues Requiring Cleaning:**
        - Negative emission values
        - Invalid percentages (>100%)
        - Extreme outliers (>10x normal)
        - Zero records (missing data)
        - Invalid dates
        
        **Click below to generate a cleaned database that all analytics pages will use.**
        """)
        
        if st.button("🚀 Generate Cleaned Database", type="primary"):
            with st.spinner("Applying data cleaning transformations..."):
                
                # Load all data
                df_raw = pd.read_sql_query("""
                SELECT e.*, f.facility_name, f.region, f.facility_type
                FROM emissions_monthly e
                JOIN facilities f ON e.facility_id = f.facility_id
                """, sqlite3.connect('../data/sustainability_data.db'))
                
                df_raw['date'] = pd.to_datetime(df_raw['date'])
                
                original_count = len(df_raw)
                cleaning_log = []
                
                # 1. Fix negative values
                negative_cols = ['scope1_tonnes', 'scope2_location_tonnes', 'scope2_market_tonnes', 'scope3_tonnes']
                negative_count = 0
                for col in negative_cols:
                    mask = df_raw[col] < 0
                    if mask.sum() > 0:
                        facility_median = df_raw.groupby('facility_id')[col].transform('median')
                        df_raw.loc[mask, col] = facility_median[mask]
                        negative_count += mask.sum()
                
                if negative_count > 0:
                    cleaning_log.append(f"✅ Fixed {negative_count} negative values (replaced with facility median)")
                
                # 2. Fix zero records with interpolation
                df_raw = df_raw.sort_values(['facility_id', 'date'])
                zero_mask = (df_raw['scope1_tonnes'] == 0) & (df_raw['scope2_market_tonnes'] == 0) & (df_raw['scope3_tonnes'] == 0)
                zero_count = zero_mask.sum()
                
                if zero_count > 0:
                    for col in negative_cols:
                        df_raw[col] = df_raw.groupby('facility_id')[col].transform(
                            lambda x: x.replace(0, np.nan).interpolate(method='linear').fillna(x.median())
                        )
                    cleaning_log.append(f"✅ Fixed {zero_count} zero records (linear interpolation)")
                
                # 3. Cap extreme outliers at 99th percentile
                extreme_count = 0
                for col in negative_cols:
                    p99 = df_raw.groupby('facility_id')[col].transform(lambda x: x.quantile(0.99))
                    mask = df_raw[col] > p99
                    if mask.sum() > 0:
                        df_raw.loc[mask, col] = p99[mask]
                        extreme_count += mask.sum()
                
                if extreme_count > 0:
                    cleaning_log.append(f"✅ Capped {extreme_count} extreme outliers (99th percentile)")
                
                # 4. Fix invalid percentages
                invalid_pct = (df_raw['renewable_pct'] > 100).sum()
                if invalid_pct > 0:
                    df_raw.loc[df_raw['renewable_pct'] > 100, 'renewable_pct'] = df_raw['renewable_pct'] / 10
                    df_raw['renewable_pct'] = df_raw['renewable_pct'].clip(0, 100)
                    cleaning_log.append(f"✅ Fixed {invalid_pct} invalid percentages (decimal errors)")
                
                # 5. Remove date outliers
                date_mask = (df_raw['date'] >= '2020-01-01') & (df_raw['date'] <= '2025-12-31')
                date_outliers = (~date_mask).sum()
                if date_outliers > 0:
                    df_raw = df_raw[date_mask]
                    cleaning_log.append(f"✅ Removed {date_outliers} records with invalid dates")
                
                # 6. Fix inconsistent Scope 2 (market > location)
                scope2_mask = df_raw['scope2_market_tonnes'] > df_raw['scope2_location_tonnes']
                scope2_issues = scope2_mask.sum()
                if scope2_issues > 0:
                    df_raw.loc[scope2_mask, 'scope2_market_tonnes'] = df_raw.loc[scope2_mask, 'scope2_location_tonnes']
                    cleaning_log.append(f"✅ Fixed {scope2_issues} inconsistent Scope 2 values")
                
                cleaned_count = len(df_raw)
                
                # Create cleaned database
                conn_clean = sqlite3.connect('../data/sustainability_data_clean.db')
                cursor_clean = conn_clean.cursor()
                
                # Copy schema from original database
                conn_orig = sqlite3.connect('../data/sustainability_data.db')
                
                # Get all table schemas
                tables_query = "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                tables = pd.read_sql_query(tables_query, conn_orig)
                
                # Create tables in cleaned database
                for _, row in tables.iterrows():
                    if row['sql']:
                        cursor_clean.execute(row['sql'])
                
                # Copy static tables (facilities, targets, etc.)
                static_tables = ['facilities', 'emission_factors', 'emission_targets', 'scope3_categories', 'business_metrics']
                for table in static_tables:
                    try:
                        df_table = pd.read_sql_query(f"SELECT * FROM {table}", conn_orig)
                        df_table.to_sql(table, conn_clean, if_exists='replace', index=False)
                    except:
                        pass  # Table might not exist
                
                # Write cleaned emissions data
                df_raw.to_sql('emissions_monthly', conn_clean, if_exists='replace', index=False)
                
                conn_clean.commit()
                conn_clean.close()
                conn_orig.close()
                
                st.success("🎉 Cleaned database created successfully!")
                
                # Show cleaning summary
                st.markdown("#### 📊 Cleaning Summary")
                
                for log in cleaning_log:
                    st.markdown(log)
                
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("Original Records", f"{original_count:,}")
                
                with col_b:
                    st.metric("Cleaned Records", f"{cleaned_count:,}")
                
                with col_c:
                    removed = original_count - cleaned_count
                    st.metric("Records Removed", f"{removed:,}", 
                             delta=f"{removed/original_count*100:.1f}%",
                             delta_color="inverse")
                
                st.info("""
                ✅ **Cleaned database created:** `sustainability_data_clean.db`
                
                **Next Steps:**
                1. Refresh this page to see improved quality scores
                2. All analytics pages will now use the cleaned data automatically
                3. Original raw data is preserved in `sustainability_data.db`
                """)
                
                st.balloons()
    
    elif use_cleaned:
        st.success("""
        ✅ **Currently using cleaned database**
        
        All quality checks are running on cleaned data. To revert to raw data:
        1. Delete `sustainability_data_clean.db`
        2. Refresh this page
        """)
        
        if st.button("🗑️ Delete Cleaned Database (Revert to Raw)"):
            try:
                os.remove('../data/sustainability_data_clean.db')
                st.success("✅ Cleaned database deleted. Refresh page to use raw data.")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error deleting database: {e}")
    
    st.markdown("---")
    
    # Export
    st.markdown("### 💾 Export Reports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        report = f"""
DATA QUALITY REPORT
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
Database: {'Cleaned' if use_cleaned else 'Raw'}

OVERALL SCORE: {overall_score:.1f}%
- Completeness: {completeness_score:.1f}%
- Consistency: {consistency_score:.1f}%
- Outliers: {outlier_count}

CONSISTENCY ISSUES:
- Negative Values: {results['consistency']['negative_values'].iloc[0]}
- Invalid Renewable %: {results['consistency']['invalid_renewable'].iloc[0]}

TIMELINESS:
- Latest Date: {latest_date.strftime('%Y-%m-%d')}
- Days Since Update: {days_old}
        """
        
        st.download_button(
            label="📄 Download Quality Report (TXT)",
            data=report,
            file_name=f"data_quality_report_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )
    
    with col2:
        if len(results['outliers']) > 0:
            outliers_csv = results['outliers'].to_csv(index=False)
            st.download_button(
                label="🚨 Download Outliers (CSV)",
                data=outliers_csv,
                file_name=f"outliers_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

except Exception as e:
    st.error(f"Error running quality checks: {e}")
    import traceback
    st.code(traceback.format_exc())

conn.close()