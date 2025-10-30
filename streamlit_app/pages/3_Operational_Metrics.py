import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
sys.path.append('..')
from utils.data_loader import load_combined_metrics, load_facilities

st.set_page_config(page_title="Operational Metrics", page_icon="⚡", layout="wide")

st.markdown('<style>.main-header {font-size: 2.5rem; font-weight: 700; color: #16a085;}</style>', unsafe_allow_html=True)
st.markdown('<p class="main-header">⚡ Operational Efficiency Metrics</p>', unsafe_allow_html=True)

st.markdown("""
## Beyond Emissions: Operational Performance

Track key operational metrics that drive sustainability performance:
- **Energy Efficiency (PUE)**: Power Usage Effectiveness in data centers
- **Carbon-Free Energy (CFE)**: Hourly renewable matching
- **Water Management**: Consumption, replenishment, and intensity
- **Waste Circularity**: Diversion rates and circular economy

**Why These Matter:**
- PUE improvements directly reduce energy consumption and emissions
- CFE percentage shows real-time decarbonization (harder than annual renewable matching)
- Water efficiency reduces costs and mitigates scarcity risk
- Waste diversion demonstrates circular economy progress
""")

st.markdown("---")

# Load data
@st.cache_data
def load_data():
    """Load operational metrics"""
    df = load_combined_metrics()
    facilities = load_facilities()
    
    # Calculate intensities
    df['emissions_per_mwh'] = (df['total_emissions'] / df['electricity_mwh']) * 1000  # kg/MWh
    df['water_per_mwh'] = df['water_consumption_gallons'] / df['electricity_mwh']
    
    return df, facilities

try:
    data, facilities = load_data()
    
    if len(data) == 0:
        st.error("No operational data available")
        st.stop()
    
    st.success(f"✅ Loaded {len(data):,} records across {data['facility_id'].nunique()} facilities")
    
    # Sidebar filters
    st.sidebar.header("🎛️ Filters")
    
    # Date range
    min_date = data['date'].min()
    max_date = data['date'].max()
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        data = data[(data['date'] >= pd.Timestamp(start_date)) & (data['date'] <= pd.Timestamp(end_date))]
    
    # Facility type filter
    facility_types = st.sidebar.multiselect(
        "Facility Type",
        options=data['facility_type'].unique(),
        default=data['facility_type'].unique()
    )
    
    data = data[data['facility_type'].isin(facility_types)]
    
    # Region filter
    regions = st.sidebar.multiselect(
        "Region",
        options=data['region'].unique(),
        default=data['region'].unique()
    )
    
    data = data[data['region'].isin(regions)]
    
    if len(data) == 0:
        st.warning("No data matches the selected filters")
        st.stop()
    
    # ====================
    # KEY METRICS
    # ====================
    st.markdown("### 📊 Current Performance")
    
    # Get latest month's data
    latest_month = data['date'].max()
    latest_data = data[data['date'] == latest_month]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        # PUE (only for data centers)
        dc_data = latest_data[latest_data['facility_type'] == 'Data Center']
        if len(dc_data) > 0:
            avg_pue = dc_data['pue'].mean()
            st.metric(
                "Fleet PUE",
                f"{avg_pue:.3f}",
                help="Power Usage Effectiveness - Lower is better (1.0 = perfect)"
            )
        else:
            st.metric("Fleet PUE", "N/A")
    
    with col2:
        avg_cfe = latest_data['cfe_pct'].mean()
        st.metric(
            "Carbon-Free Energy",
            f"{avg_cfe*100:.0f}%",
            help="Hourly match of carbon-free energy sources"
        )
    
    with col3:
        total_water = latest_data['water_consumption_gallons'].sum()
        st.metric(
            "Water Consumption",
            f"{total_water/1_000_000:.1f}M gal",
            help="Total water consumption this month"
        )
    
    with col4:
        avg_water_replen = latest_data['water_replenishment_pct'].mean()
        st.metric(
            "Water Replenishment",
            f"{avg_water_replen*100:.0f}%",
            help="Percentage of water consumption replenished"
        )
    
    with col5:
        avg_waste_div = latest_data['waste_diversion_pct'].mean()
        st.metric(
            "Waste Diversion",
            f"{avg_waste_div*100:.0f}%",
            help="Percentage of waste diverted from landfills"
        )
    
    st.markdown("---")
    
    # ====================
    # PUE ANALYSIS
    # ====================
    st.markdown("### ⚡ Power Usage Effectiveness (PUE)")
    st.markdown("""
    PUE measures data center energy efficiency: **Total Facility Energy / IT Equipment Energy**
    - **1.0** = Perfect (100% of energy goes to IT)
    - **1.10** = Industry best practice (Google fleet average: 1.09)
    - **1.5+** = Inefficient (significant opportunity for improvement)
    """)
    
    # Filter to data centers only
    dc_data = data[data['facility_type'] == 'Data Center'].copy()
    
    if len(dc_data) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # PUE trend over time
            monthly_pue = dc_data.groupby(pd.Grouper(key='date', freq='M'))['pue'].mean().reset_index()
            
            fig_pue_trend = go.Figure()
            
            fig_pue_trend.add_trace(go.Scatter(
                x=monthly_pue['date'],
                y=monthly_pue['pue'],
                mode='lines+markers',
                name='Fleet Average PUE',
                line=dict(color='#3498db', width=3),
                marker=dict(size=8)
            ))
            
            # Add industry benchmark
            fig_pue_trend.add_hline(
                y=1.09,
                line_dash='dash',
                line_color='green',
                annotation_text='Industry Best (1.09)',
                annotation_position='right'
            )
            
            fig_pue_trend.update_layout(
                title='PUE Trend Over Time',
                xaxis_title='Date',
                yaxis_title='PUE',
                height=400,
                template='plotly_white',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_pue_trend, width = 'stretch')
            
            # Calculate improvement
            if len(monthly_pue) >= 2:
                first_pue = monthly_pue['pue'].iloc[0]
                last_pue = monthly_pue['pue'].iloc[-1]
                improvement = first_pue - last_pue
                improvement_pct = (improvement / first_pue) * 100
                
                if improvement > 0:
                    st.success(f"✅ PUE improved by {improvement:.3f} ({improvement_pct:.1f}%) over the period")
                else:
                    st.info(f"ℹ️ PUE changed by {improvement:.3f} ({improvement_pct:+.1f}%)")
        
        with col2:
            # PUE by facility
            facility_pue = dc_data.groupby('facility_name')['pue'].mean().sort_values()
            
            fig_pue_facility = go.Figure()
            
            colors = ['green' if pue < 1.10 else 'orange' if pue < 1.15 else 'red' 
                     for pue in facility_pue.values]
            
            fig_pue_facility.add_trace(go.Bar(
                x=facility_pue.values,
                y=facility_pue.index,
                orientation='h',
                marker_color=colors,
                text=facility_pue.values.round(3),
                textposition='outside'
            ))
            
            fig_pue_facility.update_layout(
                title='PUE by Data Center',
                xaxis_title='PUE',
                yaxis_title='',
                height=400,
                template='plotly_white',
                showlegend=False
            )
            
            st.plotly_chart(fig_pue_facility, width = 'stretch')
        
        # PUE insights
        best_facility = facility_pue.index[0]
        best_pue = facility_pue.values[0]
        worst_facility = facility_pue.index[-1]
        worst_pue = facility_pue.values[-1]
        
        st.info(f"""
        **Best Performer:** {best_facility} (PUE: {best_pue:.3f})  
        **Opportunity:** {worst_facility} (PUE: {worst_pue:.3f})  
        **Potential Savings:** Improving {worst_facility} to {best_pue:.3f} could save ~{((worst_pue - best_pue) / worst_pue * 100):.0f}% of its energy costs
        """)
    else:
        st.warning("No data center data available for PUE analysis")
    
    st.markdown("---")
    
    # ====================
    # CARBON-FREE ENERGY
    # ====================
    st.markdown("### 🌱 Carbon-Free Energy (CFE)")
    st.markdown("""
    CFE measures the percentage of electricity matched with carbon-free sources **on an hourly basis**.
    - **Hourly matching** is harder than annual 100% renewable claims
    - Requires deep renewable PPAs, storage, or fortunate grid mix
    - Google's 2024 target: 90% CFE by 2030
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CFE trend over time
        monthly_cfe = data.groupby(pd.Grouper(key='date', freq='M'))['cfe_pct'].mean().reset_index()
        
        fig_cfe_trend = go.Figure()
        
        fig_cfe_trend.add_trace(go.Scatter(
            x=monthly_cfe['date'],
            y=monthly_cfe['cfe_pct'] * 100,
            mode='lines+markers',
            name='CFE %',
            line=dict(color='#27ae60', width=3),
            fill='tozeroy',
            fillcolor='rgba(39, 174, 96, 0.2)'
        ))
        
        # Add 100% goal
        fig_cfe_trend.add_hline(
            y=100,
            line_dash='dash',
            line_color='green',
            annotation_text='100% CFE Goal',
            annotation_position='right'
        )
        
        fig_cfe_trend.update_layout(
            title='Carbon-Free Energy Progress',
            xaxis_title='Date',
            yaxis_title='CFE %',
            height=400,
            template='plotly_white',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_cfe_trend, width = 'stretch')
    
    with col2:
        # CFE by region
        regional_cfe = data.groupby('region')['cfe_pct'].mean().sort_values(ascending=False) * 100
        
        fig_cfe_region = go.Figure()
        
        colors_cfe = ['green' if cfe > 70 else 'orange' if cfe > 40 else 'red' 
                     for cfe in regional_cfe.values]
        
        fig_cfe_region.add_trace(go.Bar(
            x=regional_cfe.values,
            y=regional_cfe.index,
            orientation='h',
            marker_color=colors_cfe,
            text=regional_cfe.values.round(0).astype(int),
            texttemplate='%{text}%',
            textposition='outside'
        ))
        
        fig_cfe_region.update_layout(
            title='CFE by Region',
            xaxis_title='CFE %',
            yaxis_title='',
            height=400,
            template='plotly_white',
            showlegend=False
        )
        
        st.plotly_chart(fig_cfe_region, width = 'stretch')
    
    # CFE insights
    current_cfe = monthly_cfe['cfe_pct'].iloc[-1] * 100
    if current_cfe >= 70:
        st.success(f"✅ Excellent CFE performance ({current_cfe:.0f}%) - among industry leaders")
    elif current_cfe >= 50:
        st.info(f"ℹ️ Good CFE progress ({current_cfe:.0f}%) - continue renewable expansion")
    else:
        st.warning(f"⚠️ CFE at {current_cfe:.0f}% - significant opportunity for renewable PPAs")
    
    st.markdown("---")
    
    # ====================
    # WATER MANAGEMENT
    # ====================
    st.markdown("### 💧 Water Management")
    st.markdown("""
    Water efficiency is critical for:
    - **Cost reduction**: Water and wastewater treatment expenses
    - **Risk mitigation**: Regulatory compliance and scarcity
    - **Community relations**: Responsible stewardship in water-stressed regions
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Water consumption trend
        monthly_water = data.groupby(pd.Grouper(key='date', freq='M')).agg({
            'water_consumption_gallons': 'sum',
            'water_replenishment_pct': 'mean'
        }).reset_index()
        
        fig_water = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_water.add_trace(
            go.Bar(
                x=monthly_water['date'],
                y=monthly_water['water_consumption_gallons'] / 1_000_000,
                name='Consumption (M gal)',
                marker_color='#3498db'
            ),
            secondary_y=False
        )
        
        fig_water.add_trace(
            go.Scatter(
                x=monthly_water['date'],
                y=monthly_water['water_replenishment_pct'] * 100,
                name='Replenishment %',
                mode='lines+markers',
                line=dict(color='#27ae60', width=3),
                marker=dict(size=8)
            ),
            secondary_y=True
        )
        
        fig_water.update_xaxes(title_text='Date')
        fig_water.update_yaxes(title_text='Consumption (M gallons)', secondary_y=False)
        fig_water.update_yaxes(title_text='Replenishment %', secondary_y=True)
        
        fig_water.update_layout(
            title='Water Consumption & Replenishment',
            height=400,
            template='plotly_white',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_water, width = 'stretch')
    
    with col2:
        # Water intensity by facility type
        water_intensity = data.groupby('facility_type').apply(
            lambda x: (x['water_consumption_gallons'].sum() / x['electricity_mwh'].sum())
        ).sort_values(ascending=False)
        
        fig_water_intensity = go.Figure()
        
        fig_water_intensity.add_trace(go.Bar(
            x=water_intensity.values,
            y=water_intensity.index,
            orientation='h',
            marker_color='#1abc9c',
            text=water_intensity.values.round(0).astype(int),
            textposition='outside'
        ))
        
        fig_water_intensity.update_layout(
            title='Water Intensity by Facility Type',
            xaxis_title='Gallons per MWh',
            yaxis_title='',
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_water_intensity, width = 'stretch')
    
    # Water replenishment progress
    current_replen = monthly_water['water_replenishment_pct'].iloc[-1] * 100
    if current_replen >= 80:
        st.success(f"✅ Excellent water replenishment ({current_replen:.0f}%) - approaching 100% goal")
    elif current_replen >= 50:
        st.info(f"ℹ️ Good progress on water replenishment ({current_replen:.0f}%)")
    else:
        st.warning(f"⚠️ Water replenishment at {current_replen:.0f}% - opportunity to expand programs")
    
    st.markdown("---")
    
    # ====================
    # WASTE CIRCULARITY
    # ====================
    st.markdown("### ♻️ Waste & Circular Economy")
    st.markdown("""
    Waste diversion demonstrates circular economy progress:
    - **Recycling**: Traditional material recovery
    - **Reuse**: Hardware components in secondary markets
    - **Repurposing**: Server parts for refurbishment
    - Goal: 90%+ diversion from landfills
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Waste diversion trend
        monthly_waste = data.groupby(pd.Grouper(key='date', freq='M')).agg({
            'waste_generated_tons': 'sum',
            'waste_diverted_tons': 'sum',
            'waste_diversion_pct': 'mean'
        }).reset_index()
        
        monthly_waste['waste_landfill_tons'] = monthly_waste['waste_generated_tons'] - monthly_waste['waste_diverted_tons']
        
        fig_waste = go.Figure()
        
        fig_waste.add_trace(go.Bar(
            x=monthly_waste['date'],
            y=monthly_waste['waste_diverted_tons'],
            name='Diverted',
            marker_color='#27ae60'
        ))
        
        fig_waste.add_trace(go.Bar(
            x=monthly_waste['date'],
            y=monthly_waste['waste_landfill_tons'],
            name='Landfill',
            marker_color='#e74c3c'
        ))
        
        fig_waste.update_layout(
            title='Waste Generation & Diversion',
            xaxis_title='Date',
            yaxis_title='Waste (tonnes)',
            barmode='stack',
            height=400,
            template='plotly_white',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_waste, width = 'stretch')
    
    with col2:
        # Diversion rate trend
        fig_diversion = go.Figure()
        
        fig_diversion.add_trace(go.Scatter(
            x=monthly_waste['date'],
            y=monthly_waste['waste_diversion_pct'] * 100,
            mode='lines+markers',
            name='Diversion Rate',
            line=dict(color='#9b59b6', width=3),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(155, 89, 182, 0.2)'
        ))
        
        # Add 90% goal
        fig_diversion.add_hline(
            y=90,
            line_dash='dash',
            line_color='green',
            annotation_text='90% Goal',
            annotation_position='right'
        )
        
        fig_diversion.update_layout(
            title='Waste Diversion Rate',
            xaxis_title='Date',
            yaxis_title='Diversion %',
            height=400,
            template='plotly_white',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_diversion, width = 'stretch')
    
    # Waste insights
    current_diversion = monthly_waste['waste_diversion_pct'].iloc[-1] * 100
    if current_diversion >= 85:
        st.success(f"✅ Excellent waste diversion ({current_diversion:.0f}%) - world-class circular economy")
    elif current_diversion >= 75:
        st.info(f"ℹ️ Good waste diversion ({current_diversion:.0f}%) - approaching best practice")
    else:
        st.warning(f"⚠️ Waste diversion at {current_diversion:.0f}% - opportunity to improve recycling programs")
    
    st.markdown("---")
    
    # ====================
    # INTEGRATED VIEW
    # ====================
    st.markdown("### 🎯 Integrated Performance Dashboard")
    
    # Create normalized scores (0-100) for radar chart
    latest_full = data[data['date'] == data['date'].max()]
    
    dc_latest = latest_full[latest_full['facility_type'] == 'Data Center']
    
    if len(dc_latest) > 0:
        # Normalize metrics (higher = better)
        pue_score = (1.5 - dc_latest['pue'].mean()) / (1.5 - 1.0) * 100  # 1.0=100%, 1.5=0%
        cfe_score = dc_latest['cfe_pct'].mean() * 100
        water_replen_score = dc_latest['water_replenishment_pct'].mean() * 100
        waste_div_score = dc_latest['waste_diversion_pct'].mean() * 100
        
        # Water efficiency (lower is better, so invert)
        water_intensity_score = max(0, 100 - (dc_latest['water_per_mwh'].mean() / 2))  # Normalize around 200 gal/MWh
        
        categories = ['PUE Efficiency', 'Carbon-Free\nEnergy', 'Water\nReplenishment', 
                     'Water\nEfficiency', 'Waste\nDiversion']
        scores = [pue_score, cfe_score, water_replen_score, water_intensity_score, waste_div_score]
        
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=scores,
            theta=categories,
            fill='toself',
            name='Current Performance',
            line_color='#3498db',
            fillcolor='rgba(52, 152, 219, 0.3)'
        ))
        
        # Add target (90% for all)
        fig_radar.add_trace(go.Scatterpolar(
            r=[90, 90, 90, 90, 90],
            theta=categories,
            fill='toself',
            name='Target (90%)',
            line_color='#27ae60',
            line_dash='dash',
            fillcolor='rgba(39, 174, 96, 0.1)'
        ))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            title='Operational Performance Scorecard',
            height=500
        )
        
        st.plotly_chart(fig_radar, width = 'stretch')
        
        # Overall score
        overall_score = sum(scores) / len(scores)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if overall_score >= 80:
                st.success(f"🌟 **Overall Performance: {overall_score:.0f}/100** - Industry Leader")
            elif overall_score >= 65:
                st.info(f"✅ **Overall Performance: {overall_score:.0f}/100** - Strong Performance")
            else:
                st.warning(f"⚠️ **Overall Performance: {overall_score:.0f}/100** - Opportunity for Improvement")
    
    st.markdown("---")
    
    # Key takeaways
    st.markdown("### 💡 Key Insights & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Strengths:**")
        
        strengths = []
        if 'pue_score' in locals() and pue_score > 80:
            strengths.append("✅ Industry-leading PUE efficiency")
        if cfe_score > 70:
            strengths.append("✅ Strong carbon-free energy progress")
        if water_replen_score > 60:
            strengths.append("✅ Good water replenishment momentum")
        if waste_div_score > 80:
            strengths.append("✅ Excellent waste diversion rates")
        
        if strengths:
            for strength in strengths:
                st.markdown(strength)
        else:
            st.markdown("Focus on building operational excellence")
    
    with col2:
        st.markdown("**Opportunities:**")
        
        opportunities = []
        if 'pue_score' in locals() and pue_score < 70:
            opportunities.append("🎯 PUE improvement programs (HVAC, containment)")
        if cfe_score < 60:
            opportunities.append("🎯 Accelerate renewable energy procurement")
        if water_replen_score < 50:
            opportunities.append("🎯 Expand water replenishment initiatives")
        if waste_div_score < 75:
            opportunities.append("🎯 Enhance recycling and circular programs")
        
        if opportunities:
            for opp in opportunities:
                st.markdown(opp)
        else:
            st.markdown("✅ Maintain excellence across all metrics")
    st.markdown("---")
    
    # ====================
    # HISTORICAL PERFORMANCE ANALYSIS
    # ====================
    st.markdown("### 📊 Historical Performance & Improvement Trends")
    st.markdown("""
    Track efficiency improvements across the portfolio over time. All metrics based on 
    actual operational data from 2020-2024.
    """)
    
    # Calculate year-over-year improvements
    yearly_metrics = data.groupby(data['date'].dt.year).agg({
        'pue': 'mean',
        'cfe_pct': 'mean',
        'water_per_mwh': 'mean',
        'waste_diversion_pct': 'mean',
        'electricity_mwh': 'sum',
        'water_consumption_gallons': 'sum'
    }).reset_index()
    
    yearly_metrics.columns = ['year', 'pue', 'cfe_pct', 'water_per_mwh', 'waste_diversion_pct', 
                              'electricity_mwh', 'water_consumption_gallons']
    
    # Filter to years with full data
    yearly_metrics = yearly_metrics[yearly_metrics['year'] <= 2024]
    
    if len(yearly_metrics) >= 2:
        # Compare first and last year
        first_year = yearly_metrics.iloc[0]
        last_year = yearly_metrics.iloc[-1]
        years_span = int(last_year['year'] - first_year['year'])
        
        st.markdown(f"#### 🎯 Portfolio Improvements ({int(first_year['year'])}-{int(last_year['year'])})")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            pue_improvement = first_year['pue'] - last_year['pue']
            pue_pct = (pue_improvement / first_year['pue']) * 100
            
            st.metric(
                "PUE Improvement",
                f"{pue_improvement:.3f}",
                delta=f"{pue_pct:.1f}%",
                help=f"From {first_year['pue']:.3f} to {last_year['pue']:.3f}"
            )
            
            if pue_improvement > 0:
                st.success("✅ Efficiency improved")
            else:
                st.info("→ Stable efficiency")
        
        with col2:
            cfe_improvement = last_year['cfe_pct'] - first_year['cfe_pct']
            cfe_pct_change = cfe_improvement * 100
            
            st.metric(
                "CFE Growth",
                f"{cfe_pct_change:+.0f}%",
                delta=f"{cfe_improvement:+.1%}",
                help=f"From {first_year['cfe_pct']*100:.0f}% to {last_year['cfe_pct']*100:.0f}%"
            )
            
            if cfe_improvement > 0:
                st.success("✅ More carbon-free")
            else:
                st.warning("⚠️ CFE declined")
        
        with col3:
            water_improvement = first_year['water_per_mwh'] - last_year['water_per_mwh']
            water_pct = (water_improvement / first_year['water_per_mwh']) * 100
            
            st.metric(
                "Water Efficiency",
                f"{water_improvement:+.0f} gal/MWh",
                delta=f"{water_pct:.1f}%",
                help="Lower is better"
            )
            
            if water_improvement > 0:
                st.success("✅ More efficient")
            else:
                st.info("→ Stable usage")
        
        with col4:
            waste_improvement = last_year['waste_diversion_pct'] - first_year['waste_diversion_pct']
            waste_pct_change = waste_improvement * 100
            
            st.metric(
                "Waste Diversion",
                f"{waste_pct_change:+.0f}%",
                delta=f"{waste_improvement:+.1%}",
                help=f"From {first_year['waste_diversion_pct']*100:.0f}% to {last_year['waste_diversion_pct']*100:.0f}%"
            )
            
            if waste_improvement > 0:
                st.success("✅ Improved")
            else:
                st.info("→ Stable")
        
        st.markdown("---")
        
        # Financial impact of improvements
        st.markdown("#### 💰 Financial Impact of Efficiency Gains")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Energy Efficiency (PUE Improvement)**")
            
            if pue_improvement > 0:
                # Calculate energy saved from PUE improvement
                avg_monthly_electricity = last_year['electricity_mwh'] / 12
                
                # Energy saved = current consumption * (PUE improvement / original PUE)
                monthly_energy_saved = avg_monthly_electricity * (pue_improvement / first_year['pue'])
                annual_energy_saved = monthly_energy_saved * 12
                
                # Cost savings at $70/MWh
                monthly_cost_savings = monthly_energy_saved * 70 / 1000  # Convert to $K
                annual_cost_savings = annual_energy_saved * 70 / 1000  # Convert to $K
                
                st.markdown(f"- **Energy Saved:** {annual_energy_saved:,.0f} MWh/year")
                st.markdown(f"- **Cost Savings:** ${annual_cost_savings:,.0f}K/year")
                st.markdown(f"- **Per Month:** ${monthly_cost_savings:.0f}K")
                
                # Emissions avoided
                emissions_avoided = annual_energy_saved * 0.35  # Average grid factor
                st.markdown(f"- **Emissions Avoided:** {emissions_avoided:,.0f} tonnes CO₂e/year")
                
                st.success(f"✅ {years_span}-year improvement delivering ${annual_cost_savings:,.0f}K annual value")
            else:
                st.info("PUE remained stable - maintaining industry-leading efficiency")
        
        with col2:
            st.markdown("**Carbon-Free Energy (CFE Increase)**")
            
            if cfe_improvement > 0:
                # Calculate emissions avoided from CFE increase
                avg_monthly_electricity = last_year['electricity_mwh'] / 12
                annual_electricity = avg_monthly_electricity * 12
                
                # Emissions avoided = electricity * grid factor * CFE improvement
                emissions_avoided = annual_electricity * 0.35 * cfe_improvement  # kg CO2/kWh
                
                # Carbon cost at $50/tonne
                carbon_value = emissions_avoided * 50 / 1000  # Convert to $K
                
                st.markdown(f"- **CFE Increase:** {cfe_improvement*100:.1f} percentage points")
                st.markdown(f"- **Emissions Avoided:** {emissions_avoided:,.0f} tonnes CO₂e/year")
                st.markdown(f"- **Carbon Value:** ${carbon_value:,.0f}K/year at $50/tonne")
                
                # Regulatory benefit
                st.markdown(f"- **EU ETS Savings:** ${emissions_avoided * 85 / 1000:,.0f}K/year at €85/tonne")
                
                st.success(f"✅ CFE growth delivering ${carbon_value:,.0f}K+ annual carbon value")
            else:
                st.info("Maintaining high CFE levels")
        
        st.markdown("---")
        
        # Best vs opportunity facilities
        st.markdown("#### 🏆 Internal Benchmarking: Best Performers vs Opportunities")
        
        # Data center comparison
        dc_comparison = data[data['facility_type'] == 'Data Center'].groupby('facility_name').agg({
            'pue': 'mean',
            'cfe_pct': 'mean',
            'water_per_mwh': 'mean',
            'electricity_mwh': 'mean'
        }).reset_index()
        
        if len(dc_comparison) > 1:
            # Best and worst PUE
            best_pue_facility = dc_comparison.loc[dc_comparison['pue'].idxmin()]
            worst_pue_facility = dc_comparison.loc[dc_comparison['pue'].idxmax()]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🥇 Best Practice Example**")
                st.markdown(f"**Facility:** {best_pue_facility['facility_name']}")
                st.markdown(f"- PUE: **{best_pue_facility['pue']:.3f}**")
                st.markdown(f"- CFE: {best_pue_facility['cfe_pct']*100:.0f}%")
                st.markdown(f"- Water Intensity: {best_pue_facility['water_per_mwh']:.0f} gal/MWh")
                st.success("Industry-leading performance")
            
            with col2:
                st.markdown("**🎯 Improvement Opportunity**")
                st.markdown(f"**Facility:** {worst_pue_facility['facility_name']}")
                st.markdown(f"- PUE: **{worst_pue_facility['pue']:.3f}**")
                st.markdown(f"- Gap: {worst_pue_facility['pue'] - best_pue_facility['pue']:.3f}")
                st.markdown(f"- CFE: {worst_pue_facility['cfe_pct']*100:.0f}%")
                
                # Potential if matched best
                pue_gap = worst_pue_facility['pue'] - best_pue_facility['pue']
                potential_energy_monthly = worst_pue_facility['electricity_mwh'] * (pue_gap / worst_pue_facility['pue'])
                potential_cost_monthly = potential_energy_monthly * 70 / 1000
                potential_annual = potential_cost_monthly * 12
                
                st.warning(f"**Potential:** ${potential_annual:.0f}K/year if matched best performer")
        
        st.markdown("---")
        
        # Trajectory projection
        st.markdown("#### 📈 Improvement Trajectory to 2030")
        
        # Fit linear trend to PUE
        if len(yearly_metrics) >= 3:
            years_array = yearly_metrics['year'].values
            pue_array = yearly_metrics['pue'].values
            
            # Linear regression
            z = np.polyfit(years_array, pue_array, 1)
            slope = z[0]
            intercept = z[1]
            
            # Project to 2030
            projected_2030 = slope * 2030 + intercept
            current_pue = last_year['pue']
            improvement_to_2030 = current_pue - projected_2030
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**PUE Trajectory:**")
                st.markdown(f"- Current (2024): **{current_pue:.3f}**")
                st.markdown(f"- Projected (2030): **{projected_2030:.3f}**")
                st.markdown(f"- Expected Improvement: {improvement_to_2030:.3f}")
                st.markdown(f"- Annual Rate: {-slope:.4f}/year")
                
                if projected_2030 < 1.08:
                    st.success("✅ On track to exceed industry best practice")
                elif projected_2030 < 1.10:
                    st.info("ℹ️ Projected to maintain industry-leading levels")
                else:
                    st.warning("⚠️ May need accelerated improvement programs")
            
            with col2:
                st.markdown("**Financial Implications (by 2030):**")
                
                if improvement_to_2030 > 0:
                    # Energy savings from continued improvement
                    avg_electricity = last_year['electricity_mwh'] / 12
                    annual_electricity = avg_electricity * 12
                    
                    energy_savings_2030 = annual_electricity * (improvement_to_2030 / current_pue)
                    cost_savings_2030 = energy_savings_2030 * 70 / 1000  # $K
                    
                    st.markdown(f"- **Energy Saved:** {energy_savings_2030:,.0f} MWh/year")
                    st.markdown(f"- **Cost Savings:** ${cost_savings_2030:,.0f}K/year")
                    st.markdown(f"- **Cumulative (2024-2030):** ${cost_savings_2030 * 6 / 2:,.0f}K")
                    
                    # Typical investment for this improvement
                    estimated_investment = cost_savings_2030 * 2.5  # Assume 2.5 year payback
                    st.markdown(f"- **Est. Investment:** ${estimated_investment:,.0f}K")
                    st.markdown(f"- **Payback:** ~2-3 years")
                    
                    st.success("✅ Strong ROI for continued efficiency investments")
                else:
                    st.info("Maintaining world-class efficiency levels")
        
        st.info("""
        **Methodology Note:**
        - All improvements calculated from actual operational data (2020-2024)
        - Energy costs based on market average of $70/MWh
        - Carbon value calculated at $50/tonne (conservative) and €85/tonne (EU ETS)
        - Investment estimates use industry-standard costs for efficiency upgrades
        - Projections use linear regression on historical trends
        """)
    
    else:
        st.warning("Need at least 2 years of data for trend analysis")

except Exception as e:
    st.error(f"Error: {e}")
    import traceback
    st.code(traceback.format_exc())

except Exception as e:
    st.error(f"Error: {e}")
    import traceback
    st.code(traceback.format_exc())