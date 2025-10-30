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


st.set_page_config(page_title="Anomaly Detection", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #dc3545;
        margin-bottom: 1rem;
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

st.markdown('<p class="main-header">🚨 Anomaly Detection</p>', unsafe_allow_html=True)

st.markdown("""
## Automated Data Quality & Anomaly Detection

Using machine learning to automatically identify unusual patterns in emissions data:
- **Isolation Forest**: ML-based outlier detection
- **Statistical Methods**: Z-score analysis for statistical outliers
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
    }[x],
    help="ML methods detect complex patterns; statistical methods find simple outliers"
)

contamination = st.sidebar.slider(
    "Expected Anomaly Rate",
    min_value=0.01,
    max_value=0.20,
    value=0.05,
    step=0.01,
    format="%.2f",
    help="Percentage of data expected to be anomalous (0.05 = 5%)"
)

zscore_threshold = st.sidebar.slider(
    "Z-Score Threshold",
    min_value=2.0,
    max_value=4.0,
    value=3.0,
    step=0.1,
    help="Standard deviations for statistical outliers"
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
        st.metric("Total Records", f"{len(data):,}")
    
    with col2:
        st.metric("Facilities", data['facility_id'].nunique())
    
    with col3:
        months = (data['date'].max() - data['date'].min()).days // 30
        st.metric("Time Period", f"{months} months")
    
    with col4:
        avg_emissions = data['total_emissions'].mean()
        st.metric("Avg Emissions", f"{avg_emissions:,.0f} tonnes")
    
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
    
    st.success("✅ Anomaly detection complete!")
    
    # Display results
    if detection_method == 'both':
        col1, col2 = st.columns(2)
        
        with col1:
            if_count = results['isolation_forest']['count']
            if_pct = (if_count / len(data) * 100) if len(data) > 0 else 0
            st.metric("🤖 Isolation Forest", if_count, f"{if_pct:.1f}% of data")
        
        with col2:
            zs_count = results['zscore']['count']
            zs_pct = (zs_count / len(data) * 100) if len(data) > 0 else 0
            st.metric("📊 Z-Score", zs_count, f"{zs_pct:.1f}% of data")
    else:
        method_key = 'isolation_forest' if detection_method == 'isolation_forest' else 'zscore'
        count = results[method_key]['count']
        pct = (count / len(data) * 100) if len(data) > 0 else 0
        st.metric("Anomalies Detected", count, f"{pct:.1f}% of data")
    
    # Choose primary method for visualization
    primary_method = 'isolation_forest' if detection_method != 'zscore' else 'zscore'
    plot_data = results[primary_method]['data']
    plot_anomalies = results[primary_method]['anomalies']
    
    # Only show visualizations if we have data
    if len(plot_data) > 0:
        st.markdown("---")
        st.subheader("📈 Anomaly Visualization")
        
        # Time series plot with anomalies
        fig = go.Figure()
        
        # Normal data points
        normal_data = plot_data[plot_data['anomaly_flag'] == 1]
        
        if len(normal_data) > 0:
            fig.add_trace(go.Scatter(
                x=normal_data['date'],
                y=normal_data['total_emissions'],
                mode='markers',
                name='Normal',
                marker=dict(size=6, color='lightblue', opacity=0.6),
                text=normal_data['facility_name'],
                hovertemplate='<b>%{text}</b><br>%{x}<br>%{y:,.0f} tonnes<extra></extra>'
            ))
        
        # Anomalies
        if len(plot_anomalies) > 0:
            fig.add_trace(go.Scatter(
                x=plot_anomalies['date'],
                y=plot_anomalies['total_emissions'],
                mode='markers',
                name='Anomaly',
                marker=dict(size=12, color='red', symbol='x', line=dict(width=2)),
                text=plot_anomalies['facility_name'],
                hovertemplate='<b>%{text}</b><br>%{x}<br>%{y:,.0f} tonnes<br><b>ANOMALY</b><extra></extra>'
            ))
        
        fig.update_layout(
            title='Emissions Over Time - Anomalies Highlighted',
            xaxis_title='Date',
            yaxis_title='Total Emissions (tonnes CO₂e)',
            hovermode='closest',
            height=500,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Additional charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Anomaly score distribution
            fig_hist = go.Figure()
            
            fig_hist.add_trace(go.Histogram(
                x=plot_data['anomaly_score'],
                nbinsx=50,
                marker_color='lightblue',
                opacity=0.7
            ))
            
            if primary_method == 'zscore':
                fig_hist.add_vline(
                    x=-zscore_threshold,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Threshold"
                )
            
            fig_hist.update_layout(
                title='Anomaly Score Distribution',
                xaxis_title='Anomaly Score',
                yaxis_title='Count',
                height=350,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_hist, width='stretch')
        
        with col2:
            # Anomalies by facility
            if len(plot_anomalies) > 0:
                anomalies_by_facility = plot_anomalies.groupby('facility_name').size().sort_values(ascending=False).head(10)
                
                fig_facility = go.Figure()
                
                fig_facility.add_trace(go.Bar(
                    x=anomalies_by_facility.values,
                    y=anomalies_by_facility.index,
                    orientation='h',
                    marker=dict(color=anomalies_by_facility.values, colorscale='Reds')
                ))
                
                fig_facility.update_layout(
                    title='Top 10 Facilities with Anomalies',
                    xaxis_title='Number of Anomalies',
                    yaxis_title='Facility',
                    height=350,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_facility, width='stretch')
            else:
                st.info("No anomalies to display by facility")
    
    # Detailed anomaly table
    if len(plot_anomalies) > 0:
        st.markdown("---")
        st.subheader("🔍 High-Priority Anomalies")
        
        # Sort by severity
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
        
        # Display table
        st.dataframe(
            top_anomalies.style.format({
                'scope1_tonnes': '{:,.0f}',
                'scope2_market_tonnes': '{:,.0f}',
                'scope3_tonnes': '{:,.0f}',
                'total_emissions': '{:,.0f}',
                'anomaly_score': '{:.3f}'
            }),
            width='stretch',
            height=400
        )
        
        # Investigation workflow
        st.markdown("---")
        st.subheader("🔧 Investigation Workflow")
        
        if len(top_anomalies) > 0:
            # Select anomaly
            selected_anomaly_idx = st.selectbox(
                "Select Anomaly to Investigate",
                range(len(top_anomalies)),
                format_func=lambda i: f"{top_anomalies.iloc[i]['date']} - {top_anomalies.iloc[i]['facility_name']} ({top_anomalies.iloc[i]['total_emissions']:,.0f} tonnes)"
            )
            
            selected = top_anomalies.iloc[selected_anomaly_idx]
            
            # Anomaly details
            st.markdown(f"""
            <div class="anomaly-high">
            <h4>📋 Selected Anomaly</h4>
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
            
            # Get facility context - SAFE ACCESS
            matching_anomaly = plot_anomalies[
                (plot_anomalies['facility_name'] == selected['facility_name']) &
                (plot_anomalies['date'] == pd.to_datetime(selected['date']))
            ]
            
            if len(matching_anomaly) > 0:
                facility_id = matching_anomaly['facility_id'].iloc[0]
                facility_history = data[data['facility_id'] == facility_id].sort_values('date')
                
                if len(facility_history) > 0:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # vs. Facility average
                        facility_avg = facility_history['total_emissions'].mean()
                        deviation = ((selected['total_emissions'] - facility_avg) / facility_avg) * 100
                        st.metric(
                            "vs. Facility Average",
                            f"{deviation:+.0f}%",
                            f"{selected['total_emissions'] - facility_avg:+,.0f} tonnes"
                        )
                    
                    with col2:
                        # vs. Prior month
                        prior_month = facility_history[
                            facility_history['date'] < pd.to_datetime(selected['date'])
                        ].tail(1)
                        
                        if len(prior_month) > 0:
                            mom_change = ((selected['total_emissions'] - prior_month['total_emissions'].iloc[0]) / 
                                          prior_month['total_emissions'].iloc[0]) * 100
                            st.metric(
                                "vs. Prior Month",
                                f"{mom_change:+.0f}%",
                                f"{selected['total_emissions'] - prior_month['total_emissions'].iloc[0]:+,.0f} tonnes"
                            )
                        else:
                            st.info("No prior month available")
                    
                    # Facility trend chart
                    fig_trend = go.Figure()
                    
                    fig_trend.add_trace(go.Scatter(
                        x=facility_history['date'],
                        y=facility_history['total_emissions'],
                        mode='lines+markers',
                        name='Historical',
                        line=dict(color='lightblue', width=2),
                        marker=dict(size=6)
                    ))
                    
                    # Highlight anomaly
                    anomaly_point = facility_history[
                        facility_history['date'] == pd.to_datetime(selected['date'])
                    ]
                    
                    if len(anomaly_point) > 0:
                        fig_trend.add_trace(go.Scatter(
                            x=anomaly_point['date'],
                            y=anomaly_point['total_emissions'],
                            mode='markers',
                            name='Anomaly',
                            marker=dict(size=20, color='red', symbol='x', line=dict(width=3))
                        ))
                    
                    # Average line
                    fig_trend.add_hline(
                        y=facility_avg,
                        line_dash="dash",
                        line_color="gray",
                        annotation_text="Facility Avg"
                    )
                    
                    fig_trend.update_layout(
                        title=f'{selected["facility_name"]} - Emissions Trend',
                        xaxis_title='Date',
                        yaxis_title='Emissions (tonnes CO₂e)',
                        height=400,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig_trend, width='stretch')
            else:
                st.warning("Could not load facility context for this anomaly")
            
            # Investigation checklist
            st.markdown("""
            ### ✅ Investigation Checklist
            
            **Data Verification:**
            - [ ] Verify data entry accuracy
            - [ ] Check utility bills match reported values
            - [ ] Confirm emission factors used
            
            **Operational Context:**
            - [ ] Check for maintenance shutdowns
            - [ ] Review production volume changes
            - [ ] Identify new equipment or processes
            
            **Resolution:**
            - [ ] Document findings
            - [ ] Correct data if error found
            - [ ] Flag for management if material
            """)
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("✅ Mark as Verified", type="primary"):
                    st.success("Marked as verified")
            
            with col2:
                if st.button("🔧 Needs Correction"):
                    st.warning("Flagged for correction")
            
            with col3:
                if st.button("🚨 Escalate"):
                    st.error("Escalated to management")
    
    else:
        st.success("✅ No anomalies detected! All data points are within normal ranges.")
        st.info("Consider adjusting detection parameters if you want more sensitive detection.")
    
    # Summary insights
    st.markdown("---")
    st.subheader("💡 Detection Summary")
    
    if len(plot_anomalies) > 0:
        anomaly_rate = len(plot_anomalies) / len(data) * 100
        facilities_with_anomalies = plot_anomalies['facility_id'].nunique()
        total_facilities = data['facility_id'].nunique()
        
        # Recent anomalies
        recent_data = data[data['date'] >= (data['date'].max() - pd.DateOffset(months=3))]
        recent_anomalies = plot_anomalies[
            plot_anomalies['date'] >= (data['date'].max() - pd.DateOffset(months=3))
        ]
        recent_rate = (len(recent_anomalies) / len(recent_data) * 100) if len(recent_data) > 0 else 0
        
        anomalies_by_facility = plot_anomalies.groupby('facility_name').size().sort_values(ascending=False)
        
        st.markdown(f"""
        **Overall Statistics:**
        - **Anomaly Rate:** {anomaly_rate:.1f}% of all data points
        - **Facilities Affected:** {facilities_with_anomalies} of {total_facilities} ({facilities_with_anomalies/total_facilities*100:.0f}%)
        - **Recent Trend (3 months):** {recent_rate:.1f}% anomaly rate
        
        **Top Risk Facilities:**
        1. {anomalies_by_facility.index[0] if len(anomalies_by_facility) > 0 else 'N/A'} - {anomalies_by_facility.iloc[0] if len(anomalies_by_facility) > 0 else 0} anomalies
        2. {anomalies_by_facility.index[1] if len(anomalies_by_facility) > 1 else 'N/A'} - {anomalies_by_facility.iloc[1] if len(anomalies_by_facility) > 1 else 0} anomalies
        3. {anomalies_by_facility.index[2] if len(anomalies_by_facility) > 2 else 'N/A'} - {anomalies_by_facility.iloc[2] if len(anomalies_by_facility) > 2 else 0} anomalies
        """)
    else:
        st.info("No anomalies detected with current settings.")
    
    # Export
    st.markdown("---")
    st.subheader("💾 Export Results")
    
    if len(plot_anomalies) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV export
            csv = plot_anomalies[['date', 'facility_id', 'facility_name', 'total_emissions', 'anomaly_score']].to_csv(index=False)
            st.download_button(
                label="📥 Download Anomalies (CSV)",
                data=csv,
                file_name=f"anomalies_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Report
            report = f"""
ANOMALY DETECTION REPORT
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

Method: {detection_method.upper()}
Contamination: {contamination:.1%}

Anomalies: {len(plot_anomalies):,}
Rate: {len(plot_anomalies)/len(data)*100:.1f}%

TOP ANOMALIES:
{chr(10).join([f"- {row['facility_name']}: {row['total_emissions']:,.0f} tonnes" for _, row in plot_anomalies.head(5).iterrows()])}
            """
            
            st.download_button(
                label="📄 Download Report (TXT)",
                data=report,
                file_name=f"anomaly_report.txt",
                mime="text/plain"
            )
    else:
        st.info("No anomalies to export")

except Exception as e:
    st.error(f"Error: {e}")
    import traceback
    st.code(traceback.format_exc())