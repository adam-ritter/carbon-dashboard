import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
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

st.markdown("---")

conn = sqlite3.connect('../data/sustainability_data.db')

@st.cache_data
def run_quality_checks():
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
    """, conn)
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
    """, conn)
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
            WHEN scope1_tonnes > (SELECT AVG(scope1_tonnes) * 3 FROM emissions_monthly) THEN 'High Scope 1'
            WHEN scope2_market_tonnes > (SELECT AVG(scope2_market_tonnes) * 3 FROM emissions_monthly) THEN 'High Scope 2'
            WHEN scope3_tonnes > (SELECT AVG(scope3_tonnes) * 3 FROM emissions_monthly) THEN 'High Scope 3'
            ELSE 'Normal'
        END as flag
    FROM emissions_monthly
    WHERE scope1_tonnes > (SELECT AVG(scope1_tonnes) * 3 FROM emissions_monthly)
       OR scope2_market_tonnes > (SELECT AVG(scope2_market_tonnes) * 3 FROM emissions_monthly)
       OR scope3_tonnes > (SELECT AVG(scope3_tonnes) * 3 FROM emissions_monthly)
    """, conn)
    results['outliers'] = range_df
    
    # Timeliness check
    timeliness_df = pd.read_sql_query("""
    SELECT 
        MAX(date) as latest_date,
        julianday('now') - julianday(MAX(date)) as days_since_last_update,
        COUNT(DISTINCT facility_id) as facilities_reporting
    FROM emissions_monthly
    """, conn)
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
    """, conn)
    results['continuity'] = continuity_df
    
    return results

try:
    st.markdown("### 🔍 Running Quality Checks...")
    with st.spinner("Analyzing data quality..."):
        results = run_quality_checks()
    
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
        title={'text': "Data Quality Score"},
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
    st.plotly_chart(fig_gauge, width = 'stretch'
    
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
    
    st.plotly_chart(fig_complete, width = 'stretch'
    
    st.dataframe(completeness_display, width = 'stretch'
    
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
    
    st.plotly_chart(fig_consistency, width = 'stretch'
    
    st.dataframe(consistency_summary, width = 'stretch'
    
    if consistency_issues > 0:
        st.warning(f"⚠️ Found {consistency_issues} consistency issues requiring investigation")
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
        st.plotly_chart(fig_outliers, width = 'stretch'
        
        st.dataframe(outliers_display.head(20), width = 'stretch'
        
        st.info("""
        **Recommended Actions:**
        - Review flagged records with facility managers
        - Verify data entry accuracy
        - Investigate operational changes
        - Document legitimate outliers
        """)
    else:
        st.success("✅ No outliers detected - data within expected ranges")
    
    st.markdown("---")
    
    # Timeliness
    st.markdown("### ⏰ Data Timeliness")
    
    timeliness = results['timeliness'].iloc[0]
    latest_date = pd.to_datetime(timeliness['latest_date'])
    days_old = int(timeliness['days_since_last_update'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Latest Data Date",
            latest_date.strftime('%Y-%m-%d'),
            f"{days_old} days ago"
        )
    
    with col2:
        st.metric(
            "Facilities Reporting",
            int(timeliness['facilities_reporting']),
            help="Number of facilities with data"
        )
    
    if days_old > 60:
        st.error(f"⚠️ Data is {days_old} days old - update needed for accurate reporting")
    elif days_old > 30:
        st.warning(f"⚡ Data is {days_old} days old - consider updating soon")
    else:
        st.success(f"✅ Data is current (last updated {days_old} days ago)")
    
    st.markdown("---")
    
    # Continuity check
    st.markdown("### 📅 Monthly Reporting Continuity")
    st.markdown("Facilities with incomplete monthly data.")
    
    if len(results['continuity']) > 0:
        st.warning(f"⚠️ {len(results['continuity'])} facilities have gaps in monthly reporting")
        
        continuity_display = results['continuity'].copy()
        
        fig_continuity = go.Figure(go.Bar(
            x=continuity_display['completeness_pct'],
            y=continuity_display['facility_name'],
            orientation='h',
            marker=dict(
                color=continuity_display['completeness_pct'],
                colorscale='RdYlGn',
                cmin=0,
                cmax=100
            ),
            text=continuity_display['completeness_pct'].apply(lambda x: f"{x:.0f}%"),
            textposition='outside'
        ))
        
        fig_continuity.update_layout(
            title='Reporting Completeness by Facility',
            xaxis_title='% of Expected Months Reported',
            height=max(300, len(continuity_display) * 25),
            template='plotly_white',
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig_continuity, width = 'stretch'
        st.dataframe(continuity_display, width = 'stretch'
    else:
        st.success("✅ All facilities have complete monthly reporting!")
    
    st.markdown("---")
    
    # Recommendations
    st.markdown("### 💡 Quality Improvement Recommendations")
    
    recommendations = []
    
    if completeness_score < 95:
        recommendations.append("🔴 **Completeness:** Implement mandatory field validation at data entry")
    
    if consistency_issues > total_records * 0.01:
        recommendations.append("🔴 **Consistency:** Add automated validation rules before database commit")
    
    if outlier_count > total_records * 0.05:
        recommendations.append("🟡 **Outliers:** Review and document legitimate high-emission events")
    
    if days_old > 30:
        recommendations.append("🟡 **Timeliness:** Establish monthly data collection deadline")
    
    if len(results['continuity']) > 0:
        recommendations.append("🟡 **Continuity:** Follow up with facilities missing monthly reports")
    
    if not recommendations:
        recommendations.append("✅ **Excellent:** Data quality meets all standards - maintain current practices")
    
    for rec in recommendations:
        st.markdown(rec)
    
    st.markdown("---")
    
    # Export
    st.markdown("### 💾 Export Quality Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        report = f"""
DATA QUALITY REPORT
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

OVERALL SCORE: {overall_score:.1f}%
- Completeness: {completeness_score:.1f}%
- Consistency: {consistency_score:.1f}%
- Outliers: {outlier_count} detected

COMPLETENESS CHECKS:
{completeness_display.to_string(index=False)}

CONSISTENCY ISSUES:
- Inconsistent Scope 2: {results['consistency']['inconsistent_scope2'].iloc[0]}
- Negative Values: {results['consistency']['negative_values'].iloc[0]}
- Invalid Renewable %: {results['consistency']['invalid_renewable'].iloc[0]}

TIMELINESS:
- Latest Date: {latest_date.strftime('%Y-%m-%d')}
- Days Since Update: {days_old}

RECOMMENDATIONS:
{chr(10).join(recommendations)}
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