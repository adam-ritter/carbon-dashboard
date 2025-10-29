import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
sys.path.append('..')
from utils.ml_models import AnomalyDetector
import sqlite3


st.set_page_config(page_title = "Anomaly Detection", layout = "wide")

# custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #dc3545;
        margin-bottom: 1rem;
    }
    .alert-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .anomaly-high {
        background-color: #ffe6e6;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .anomaly-medium {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
## Automated Data Quality & Anomaly Detection

Using machine learning to automatically identify unusual patterns in emissions data:
- **Isolation Forest**: ML-based outlier detection
- **Statistical Methods**: Z-score analysis for statistical outliers
- **Time Series Patterns**: Detect breaks in seasonal patterns
- **Multi-dimensional**: Analyze across facilities, scopes, and time

**Business Value:** Catch data errors early, identify operational issues, ensure reporting accuracy.
""")

# Sidebar controls
st.sidebar.header("🎛️ Detection Settings")

detection_method = st.sidebar.selectbox(
    "Detection Method",
    ['isolation_forest', 'zscore', 'both'],
    format_func=lambda x: {
        'isolation_forest': 'Isolation Forest (ML)',
        'zscore': 'Z-Score (Statistical)',
        'both': 'Both Methods'
    }[x]
)

contamination = st.sidebar.slider(
    "Expected Anomaly Rate",
    min_value=0.01,
    max_value=0.20,
    value=0.05,
    step=0.01,
    format="%.0f%%",
    help="Percentage of data expected to be anomalous"
)

zscore_threshold = st.sidebar.slider(
    "Z-Score Threshold",
    min_value=2.0,
    max_value=4.0,
    value=3.0,
    step=0.1,
    help="Standard deviations for statistical outliers"
)

scope_filter = st.sidebar.multiselect(
    "Scope Filter",
    ['Scope 1', 'Scope 2', 'Scope 3'],
    default=['Scope 1', 'Scope 2', 'Scope 3']
)

# Load data
@st.cache_data
def load_facility_data():
    """Load detailed facility-level emissions data"""
    conn = sqlite3.connect('../data/sustainability_data.db')
    
    query = """
    SELECT 
        e.date,
        e.facility_id,
        f.facility_name,
        f.region,
        f.facility_type,
        e.scope1_tonnes,
        e.scope2_market_tonnes,
        e.scope3_tonnes,
        (e.scope1_tonnes + e.scope2_market_tonnes + e.scope3_tonnes) as total_emissions,
        e.electricity_mwh,
        e.renewable_pct
    FROM emissions_monthly e
    JOIN facilities f ON e.facility_id = f.facility_id
    ORDER BY e.date, e.facility_id
    """
    
    df = pd.read_sql_query(query, conn)
    df['date'] = pd.to_datetime(df['date'])
    conn.close()
    
    return df

try:
    data = load_facility_data()
    
    # Summary metrics
    st.subheader("📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Data Points",
            f"{len(data):,}",
            help="Total monthly facility records"
        )
    
    with col2:
        st.metric(
            "Facilities",
            data['facility_id'].nunique(),
            help="Number of unique facilities"
        )
    
    with col3:
        st.metric(
            "Time Period",
            f"{(data['date'].max() - data['date'].min()).days // 30} months",
            help="Months of historical data"
        )
    
    with col4:
        avg_emissions = data['total_emissions'].mean()
        st.metric(
            "Avg Emissions",
            f"{avg_emissions:,.0f} tonnes",
            help="Average monthly emissions per facility"
        )
    
    st.markdown("---")
    
    # Run anomaly detection
    st.subheader("🔍 Anomaly Detection Analysis")
    
    # Select features for detection
    detection_features = ['scope1_tonnes', 'scope2_market_tonnes', 'scope3_tonnes', 'total_emissions']
    
    with st.spinner("Running anomaly detection algorithms..."):
        
        results = {}
        
        if detection_method in ['isolation_forest', 'both']:
            # Isolation Forest
            detector_if = AnomalyDetector(method='isolation_forest', contamination=contamination)
            data_if = detector_if.fit_predict(data.copy(), detection_features)
            anomalies_if = detector_if.get_anomalies(data_if)
            results['isolation_forest'] = {
                'data': data_if,
                'anomalies': anomalies_if,
                'count': len(anomalies_if)
            }
        
        if detection_method in ['zscore', 'both']:
            # Z-score
            detector_zs = AnomalyDetector(method='zscore')
            data_zs = detector_zs.fit_predict(data.copy(), detection_features)
            # Filter by threshold
            anomalies_zs = data_zs[data_zs['anomaly_score'] < -zscore_threshold]
            results['zscore'] = {
                'data': data_zs,
                'anomalies': anomalies_zs,
                'count': len(anomalies_zs)
            }
    
    # Display results
    if detection_method == 'both':
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "🤖 Isolation Forest Anomalies",
                results['isolation_forest']['count'],
                delta=f"{results['isolation_forest']['count']/len(data)*100:.1f}% of data"
            )
        
        with col2:
            st.metric(
                "📊 Z-Score Anomalies",
                results['zscore']['count'],
                delta=f"{results['zscore']['count']/len(data)*100:.1f}% of data"
            )
    else:
        method_key = 'isolation_forest' if detection_method == 'isolation_forest' else 'zscore'
        st.metric(
            f"Anomalies Detected ({detection_method})",
            results[method_key]['count'],
            delta=f"{results[method_key]['count']/len(data)*100:.1f}% of data"
        )
    
    # Visualization
    st.subheader("📈 Anomaly Visualization")
    
    # Choose primary method for visualization
    primary_method = 'isolation_forest' if detection_method != 'zscore' else 'zscore'
    plot_data = results[primary_method]['data']
    plot_anomalies = results[primary_method]['anomalies']
    
    # Time series plot with anomalies
    fig = go.Figure()
    
    # Normal data points
    normal_data = plot_data[plot_data['anomaly_flag'] == 1]
    
    fig.add_trace(go.Scatter(
        x=normal_data['date'],
        y=normal_data['total_emissions'],
        mode='markers',
        name='Normal',
        marker=dict(
            size=6,
            color='lightblue',
            opacity=0.6,
            line=dict(width=0.5, color='darkblue')
        ),
        text=normal_data['facility_name'],
        hovertemplate='<b>%{text}</b><br>Date: %{x}<br>Emissions: %{y:,.0f} tonnes<extra></extra>'
    ))
    
    # Anomalies
    fig.add_trace(go.Scatter(
        x=plot_anomalies['date'],
        y=plot_anomalies['total_emissions'],
        mode='markers',
        name='Anomaly',
        marker=dict(
            size=12,
            color='red',
            symbol='x',
            line=dict(width=2, color='darkred')
        ),
        text=plot_anomalies['facility_name'],
        hovertemplate='<b>%{text}</b><br>Date: %{x}<br>Emissions: %{y:,.0f} tonnes<br><b>ANOMALY</b><extra></extra>'
    ))
    
    fig.update_layout(
        title='Emissions Over Time - Anomalies Highlighted',
        xaxis_title='Date',
        yaxis_title='Total Emissions (tonnes CO₂e)',
        hovermode='closest',
        height=500,
        template='plotly_white',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Anomaly score distribution
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram of anomaly scores
        fig_hist = go.Figure()
        
        fig_hist.add_trace(go.Histogram(
            x=plot_data['anomaly_score'],
            nbinsx=50,
            name='Anomaly Score Distribution',
            marker_color='lightblue',
            opacity=0.7
        ))
        
        # Add threshold line
        if primary_method == 'zscore':
            fig_hist.add_vline(
                x=-zscore_threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Threshold ({zscore_threshold}σ)"
            )
        
        fig_hist.update_layout(
            title='Anomaly Score Distribution',
            xaxis_title='Anomaly Score',
            yaxis_title='Count',
            height=350,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Anomalies by facility
        anomalies_by_facility = plot_anomalies.groupby('facility_name').size().sort_values(ascending=False)
        
        fig_facility = go.Figure()
        
        fig_facility.add_trace(go.Bar(
            x=anomalies_by_facility.values,
            y=anomalies_by_facility.index,
            orientation='h',
            marker=dict(
                color=anomalies_by_facility.values,
                colorscale='Reds',
                showscale=False
            )
        ))
        
        fig_facility.update_layout(
            title='Anomalies by Facility',
            xaxis_title='Number of Anomalies',
            yaxis_title='Facility',
            height=350,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_facility, use_container_width=True)
    
    # Detailed anomaly table
    st.subheader("🔍 High-Priority Anomalies")
    
    # Sort by severity (most anomalous first)
    top_anomalies = plot_anomalies.nsmallest(20, 'anomaly_score')[[
        'date', 'facility_name', 'region', 'facility_type',
        'scope1_tonnes', 'scope2_market_tonnes', 'scope3_tonnes',
        'total_emissions', 'anomaly_score'
    ]].copy()
    
    top_anomalies['date'] = top_anomalies['date'].dt.strftime('%Y-%m')
    top_anomalies['severity'] = pd.cut(
        -top_anomalies['anomaly_score'],
        bins=3,
        labels=['Medium', 'High', 'Critical']
    )
    
    # Color coding
    def highlight_severity(row):
        if row['severity'] == 'Critical':
            return ['background-color: #ffe6e6'] * len(row)
        elif row['severity'] == 'High':
            return ['background-color: #fff3cd'] * len(row)
        else:
            return ['background-color: #e7f3ff'] * len(row)
    
    st.dataframe(
        top_anomalies.style
            .format({
                'scope1_tonnes': '{:,.0f}',
                'scope2_market_tonnes': '{:,.0f}',
                'scope3_tonnes': '{:,.0f}',
                'total_emissions': '{:,.0f}',
                'anomaly_score': '{:.3f}'
            })
            .apply(highlight_severity, axis=1),
        use_container_width=True,
        height=400
    )
    
    # Investigation workflow
    st.subheader("🔧 Investigation Workflow")
    
    if len(plot_anomalies) > 0:
        # Select anomaly for investigation
        selected_anomaly_idx = st.selectbox(
            "Select Anomaly to Investigate",
            range(len(top_anomalies)),
            format_func=lambda i: f"{top_anomalies.iloc[i]['date']} - {top_anomalies.iloc[i]['facility_name']} ({top_anomalies.iloc[i]['total_emissions']:,.0f} tonnes)"
        )
        
        selected = top_anomalies.iloc[selected_anomaly_idx]
        
        # Investigation details
        st.markdown(f"""
        <div class="anomaly-high">
        <h4>📋 Anomaly Details</h4>
        <ul>
            <li><strong>Date:</strong> {selected['date']}</li>
            <li><strong>Facility:</strong> {selected['facility_name']} ({selected['facility_type']})</li>
            <li><strong>Region:</strong> {selected['region']}</li>
            <li><strong>Total Emissions:</strong> {selected['total_emissions']:,.0f} tonnes CO₂e</li>
            <li><strong>Anomaly Score:</strong> {selected['anomaly_score']:.3f}</li>
            <li><strong>Severity:</strong> {selected['severity']}</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Context comparison
        facility_id = plot_anomalies[
            (plot_anomalies['facility_name'] == selected['facility_name']) &
            (plot_anomalies['date'] == pd.to_datetime(selected['date']))
        ]['facility_id'].iloc[0]
        
        facility_history = data[data['facility_id'] == facility_id].sort_values('date')
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Compare to facility average
            facility_avg = facility_history['total_emissions'].mean()
            deviation = ((selected['total_emissions'] - facility_avg) / facility_avg) * 100
            
            st.metric(
                "vs. Facility Average",
                f"{deviation:+.0f}%",
                delta=f"{selected['total_emissions'] - facility_avg:+,.0f} tonnes"
            )
        
        with col2:
            # Compare to prior month
            prior_month = facility_history[
                facility_history['date'] < pd.to_datetime(selected['date'])
            ].tail(1)
            
            if len(prior_month) > 0:
                mom_change = ((selected['total_emissions'] - prior_month['total_emissions'].iloc[0]) / 
                              prior_month['total_emissions'].iloc[0]) * 100
                st.metric(
                    "vs. Prior Month",
                    f"{mom_change:+.0f}%",
                    delta=f"{selected['total_emissions'] - prior_month['total_emissions'].iloc[0]:+,.0f} tonnes"
                )
        
        # Facility trend
        fig_trend = go.Figure()
        
        fig_trend.add_trace(go.Scatter(
            x=facility_history['date'],
            y=facility_history['total_emissions'],
            mode='lines+markers',
            name='Historical Trend',
            line=dict(color='lightblue', width=2),
            marker=dict(size=6)
        ))
        
        # Highlight anomalous month
        anomaly_point = facility_history[
            facility_history['date'] == pd.to_datetime(selected['date'])
        ]
        
        fig_trend.add_trace(go.Scatter(
            x=anomaly_point['date'],
            y=anomaly_point['total_emissions'],
            mode='markers',
            name='Anomaly',
            marker=dict(size=20, color='red', symbol='x', line=dict(width=3))
        ))
        
        # Add average line
        fig_trend.add_hline(
            y=facility_avg,
            line_dash="dash",
            line_color="gray",
            annotation_text="Facility Average"
        )
        
        fig_trend.update_layout(
            title=f'{selected["facility_name"]} - Emissions Trend',
            xaxis_title='Date',
            yaxis_title='Total Emissions (tonnes CO₂e)',
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Investigation checklist
        st.markdown("""
        ### ✅ Investigation Checklist
        
        **Data Verification:**
        - [ ] Verify data entry accuracy
        - [ ] Check utility bills match reported values
        - [ ] Confirm emission factors used
        - [ ] Review calculation methodology
        
        **Operational Context:**
        - [ ] Check for maintenance shutdowns
        - [ ] Review production volume changes
        - [ ] Identify new equipment or processes
        - [ ] Check for weather anomalies (heating/cooling)
        
        **Resolution:**
        - [ ] Document findings
        - [ ] Correct data if error found
        - [ ] Update procedures if systemic issue
        - [ ] Flag for senior management if material
        """)
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("✅ Mark as Verified", type="primary"):
                st.success("Anomaly marked as verified and explainable")
        
        with col2:
            if st.button("🔧 Data Correction Needed"):
                st.warning("Flagged for data correction")
        
        with col3:
            if st.button("🚨 Escalate to Management"):
                st.error("Escalated for management review")
    
    # Summary insights
    st.subheader("💡 Detection Summary & Insights")
    
    # Calculate statistics
    anomaly_rate = len(plot_anomalies) / len(data) * 100
    facilities_with_anomalies = plot_anomalies['facility_id'].nunique()
    total_facilities = data['facility_id'].nunique()
    
    # Time period analysis
    recent_anomalies = plot_anomalies[
        plot_anomalies['date'] >= (data['date'].max() - pd.DateOffset(months=3))
    ]
    recent_rate = len(recent_anomalies) / len(
        data[data['date'] >= (data['date'].max() - pd.DateOffset(months=3))]
    ) * 100 if len(data[data['date'] >= (data['date'].max() - pd.DateOffset(months=3))]) > 0 else 0
    
    st.markdown(f"""
    ### 📊 Key Findings
    
    **Overall Statistics:**
    - **Anomaly Rate:** {anomaly_rate:.1f}% of all data points
    - **Facilities Affected:** {facilities_with_anomalies} of {total_facilities} facilities ({facilities_with_anomalies/total_facilities*100:.0f}%)
    - **Recent Trend (3 months):** {recent_rate:.1f}% anomaly rate
    - **Change:** {'Increasing' if recent_rate > anomaly_rate else 'Decreasing'} anomaly frequency
    
    **Top Risk Areas:**
    1. {anomalies_by_facility.index[0] if len(anomalies_by_facility) > 0 else 'N/A'} - {anomalies_by_facility.iloc[0] if len(anomalies_by_facility) > 0 else 0} anomalies
    2. {anomalies_by_facility.index[1] if len(anomalies_by_facility) > 1 else 'N/A'} - {anomalies_by_facility.iloc[1] if len(anomalies_by_facility) > 1 else 0} anomalies
    3. {anomalies_by_facility.index[2] if len(anomalies_by_facility) > 2 else 'N/A'} - {anomalies_by_facility.iloc[2] if len(anomalies_by_facility) > 2 else 0} anomalies
    
    **Recommended Actions:**
    - {"High priority" if anomaly_rate > 10 else "Standard"} review of flagged data points
    - {"Implement" if facilities_with_anomalies/total_facilities > 0.5 else "Continue"} enhanced data quality controls
    - {"Urgent" if recent_rate > anomaly_rate * 1.5 else "Routine"} investigation of recent anomalies
    """)
    
    # Export functionality
    st.subheader("💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export anomalies CSV
        anomalies_export = plot_anomalies[[
            'date', 'facility_id', 'facility_name', 'region', 'facility_type',
            'scope1_tonnes', 'scope2_market_tonnes', 'scope3_tonnes',
            'total_emissions', 'anomaly_score'
        ]].copy()
        
        csv = anomalies_export.to_csv(index=False)
        st.download_button(
            label="📥 Download Anomalies (CSV)",
            data=csv,
            file_name=f"anomalies_{detection_method}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Export investigation report
        report = f"""
ANOMALY DETECTION REPORT
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

METHOD: {detection_method.upper()}
Settings:
- Contamination: {contamination:.1%}
- Z-Score Threshold: {zscore_threshold if detection_method != 'isolation_forest' else 'N/A'}

SUMMARY:
- Total Data Points: {len(data):,}
- Anomalies Detected: {len(plot_anomalies):,} ({anomaly_rate:.1f}%)
- Facilities Affected: {facilities_with_anomalies}/{total_facilities}
- Recent Anomaly Rate (3mo): {recent_rate:.1f}%

TOP FACILITIES WITH ANOMALIES:
{chr(10).join([f"{i+1}. {name}: {count} anomalies" for i, (name, count) in enumerate(anomalies_by_facility.head(5).items())])}

HIGH-PRIORITY ANOMALIES FOR INVESTIGATION:
{chr(10).join([f"- {row['date']}: {row['facility_name']} ({row['total_emissions']:,.0f} tonnes, score: {row['anomaly_score']:.3f})" for _, row in top_anomalies.head(10).iterrows()])}

RECOMMENDATIONS:
- Review and verify all high-priority anomalies
- Investigate facilities with multiple anomalies
- Update data collection procedures for repeat offenders
- Schedule follow-up analysis in 30 days
        """
        
        st.download_button(
            label="📄 Download Report (TXT)",
            data=report,
            file_name=f"anomaly_report_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )

except Exception as e:
    st.error(f"Error loading data or running anomaly detection: {e}")
    st.info("Please ensure the database exists and contains facility-level emissions data.")
    import traceback
    st.code(traceback.format_exc())

