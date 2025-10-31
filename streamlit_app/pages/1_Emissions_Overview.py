import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
sys.path.append('..')
from utils.data_loader import load_emissions_data, load_facilities
import sys
sys.path.append('..')

st.set_page_config(page_title="Emissions Overview", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">📊 Emissions Overview</p>', unsafe_allow_html=True)

st.markdown("""
## Executive Dashboard

Real-time view of Google's emissions across all facilities and scopes. This dashboard provides 
the high-level KPIs used in board presentations and investor disclosures.
""")

# Load data
@st.cache_data
def get_overview_data():
    emissions_df = load_emissions_data()
    facilities_df = load_facilities()
    
    emissions_df['date'] = pd.to_datetime(emissions_df['date'])
    emissions_df['total_emissions'] = (
        emissions_df['scope1_tonnes'] + 
        emissions_df['scope2_market_tonnes'] + 
        emissions_df['scope3_tonnes']
    )
    emissions_df['year'] = emissions_df['date'].dt.year
    emissions_df['month'] = emissions_df['date'].dt.to_period('M').astype(str)
    
    return emissions_df, facilities_df

try:
    emissions_df, facilities_df = get_overview_data()
    
    # Sidebar filters
    st.sidebar.header("🎛️ Filters")
    
    years = sorted(emissions_df['year'].unique())
    selected_years = st.sidebar.multiselect(
        "Select Years",
        years,
        default=years
    )
    
    facilities = sorted(emissions_df['facility_id'].unique())
    selected_facilities = st.sidebar.multiselect(
        "Select Facilities",
        facilities,
        default=facilities
    )
    
    # Filter data
    filtered_df = emissions_df[
        (emissions_df['year'].isin(selected_years)) &
        (emissions_df['facility_id'].isin(selected_facilities))
    ]
    
    if len(filtered_df) == 0:
        st.warning("No data available for selected filters")
        st.stop()
    
    # Key Metrics
    st.markdown("### 🎯 Key Performance Indicators")
    
    # Calculate metrics
    latest_year = filtered_df['year'].max()
    latest_year_data = filtered_df[filtered_df['year'] == latest_year]
    
    total_emissions = latest_year_data['total_emissions'].sum()
    total_scope1 = latest_year_data['scope1_tonnes'].sum()
    total_scope2 = latest_year_data['scope2_market_tonnes'].sum()
    total_scope3 = latest_year_data['scope3_tonnes'].sum()
    avg_renewable = latest_year_data['renewable_pct'].mean()
    
    # YoY comparison
    if latest_year - 1 in filtered_df['year'].values:
        prior_year_data = filtered_df[filtered_df['year'] == latest_year - 1]
        prior_total = prior_year_data['total_emissions'].sum()
        yoy_change = ((total_emissions - prior_total) / prior_total) * 100
    else:
        yoy_change = 0
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            f"{latest_year} Total Emissions",
            f"{total_emissions/1_000_000:.2f}M tonnes",
            f"{yoy_change:+.1f}% YoY",
            delta_color="inverse"
        )
    
    with col2:
        scope3_pct = (total_scope3 / total_emissions) * 100
        st.metric(
            "Scope 3 Share",
            f"{scope3_pct:.0f}%",
            help="Scope 3 (value chain) as % of total"
        )
    
    with col3:
        st.metric(
            "Avg Renewable Energy",
            f"{avg_renewable:.0f}%",
            help="Average renewable energy % across facilities"
        )
    
    with col4:
        facility_count = len(selected_facilities)
        st.metric(
            "Facilities Analyzed",
            facility_count,
            help="Number of facilities in current view"
        )
    
    st.markdown("---")
    
    # Monthly Trend
    st.markdown("### 📈 Monthly Emissions Trend")
    
    monthly_totals = filtered_df.groupby('month').agg({
        'scope1_tonnes': 'sum',
        'scope2_market_tonnes': 'sum',
        'scope3_tonnes': 'sum',
        'total_emissions': 'sum'
    }).reset_index()
    
    # Sort by date
    monthly_totals = monthly_totals.sort_values('month')
    
    fig_trend = go.Figure()
    
    fig_trend.add_trace(go.Scatter(
        x=monthly_totals['month'],
        y=monthly_totals['total_emissions'],
        mode='lines+markers',
        name='Total Emissions',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=6),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.1)'
    ))
    
    # Add 3-month moving average
    monthly_totals['ma3'] = monthly_totals['total_emissions'].rolling(window=3).mean()
    
    fig_trend.add_trace(go.Scatter(
        x=monthly_totals['month'],
        y=monthly_totals['ma3'],
        mode='lines',
        name='3-Month Moving Avg',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))
    
    fig_trend.update_layout(
        title='Total Emissions Over Time',
        xaxis_title='Month',
        yaxis_title='Emissions (tonnes CO₂e)',
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_trend, width = 'stretch')
    
    st.markdown("---")
    
    # Scope Breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔬 Emissions by Scope")
        
        scope_data = pd.DataFrame({
            'Scope': ['Scope 1', 'Scope 2', 'Scope 3'],
            'Emissions': [total_scope1, total_scope2, total_scope3],
            'Percentage': [
                (total_scope1/total_emissions)*100,
                (total_scope2/total_emissions)*100,
                (total_scope3/total_emissions)*100
            ]
        })
        
        fig_scope = go.Figure(data=[go.Pie(
            labels=scope_data['Scope'],
            values=scope_data['Emissions'],
            hole=0.4,
            marker=dict(colors=['#ff7f0e', '#2ca02c', '#d62728']),
            textinfo='label+percent',
            textfont_size=14
        )])
        
        fig_scope.update_layout(
            title=f'Total: {total_emissions:,.0f} tonnes CO₂e',
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig_scope, width = 'stretch')
        
        # Scope metrics
        for _, row in scope_data.iterrows():
            st.metric(
                row['Scope'],
                f"{row['Emissions']/1000:,.0f}K tonnes",
                f"{row['Percentage']:.1f}% of total"
            )
    
    with col2:
        st.markdown("### 🏭 Top 10 Emitting Facilities")
        
        facility_totals = latest_year_data.groupby('facility_id').agg({
            'total_emissions': 'sum'
        }).reset_index()
        
        facility_totals = facility_totals.merge(
            facilities_df[['facility_id', 'facility_name', 'region']],
            on='facility_id'
        )
        
        facility_totals = facility_totals.sort_values('total_emissions', ascending=False).head(10)
        
        fig_facilities = go.Figure(go.Bar(
            x=facility_totals['total_emissions'],
            y=facility_totals['facility_name'],
            orientation='h',
            marker=dict(
                color=facility_totals['total_emissions'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Emissions")
            ),
            text=facility_totals['total_emissions'].apply(lambda x: f"{x/1000:.1f}K"),
            textposition='outside'
        ))
        
        fig_facilities.update_layout(
            title='Facilities Ranked by Total Emissions',
            xaxis_title='Emissions (tonnes CO₂e)',
            yaxis_title='',
            height=400,
            template='plotly_white',
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig_facilities, width = 'stretch')
    
    st.markdown("---")
    
    # Regional Breakdown
    st.markdown("### 🌍 Regional Analysis")
    
    # Debug: Check what columns we have
    if 'region' not in latest_year_data.columns:
        # Merge with facilities to get region
        regional_df = latest_year_data.merge(
            facilities_df[['facility_id', 'region']],
            on='facility_id',
            how='left'
        )
    else:
        regional_df = latest_year_data.copy()

    # Check if merge worked
    if 'region' not in regional_df.columns:
        st.error("❌ Region data not available. Check that facilities table has 'region' column.")
        st.write("Available columns:", regional_df.columns.tolist())
        st.write("Facilities columns:", facilities_df.columns.tolist())
        st.stop()

    # Check for any null regions
    if regional_df['region'].isnull().any():
        st.warning(f"⚠️ {regional_df['region'].isnull().sum()} records missing region data")
        regional_df = regional_df.dropna(subset=['region'])
    
    regional_totals = regional_df.groupby('region').agg({
        'scope1_tonnes': 'sum',
        'scope2_market_tonnes': 'sum',
        'scope3_tonnes': 'sum',
        'total_emissions': 'sum',
        'facility_id': 'count'
    }).reset_index()
    
    regional_totals.columns = ['Region', 'Scope 1', 'Scope 2', 'Scope 3', 'Total', 'Facility Count']
    
    fig_regional = go.Figure()
    
    fig_regional.add_trace(go.Bar(
        name='Scope 1',
        x=regional_totals['Region'],
        y=regional_totals['Scope 1'],
        marker_color='#ff7f0e'
    ))
    
    fig_regional.add_trace(go.Bar(
        name='Scope 2',
        x=regional_totals['Region'],
        y=regional_totals['Scope 2'],
        marker_color='#2ca02c'
    ))
    
    fig_regional.add_trace(go.Bar(
        name='Scope 3',
        x=regional_totals['Region'],
        y=regional_totals['Scope 3'],
        marker_color='#d62728'
    ))
    
    fig_regional.update_layout(
        title='Emissions by Region (Stacked)',
        xaxis_title='Region',
        yaxis_title='Emissions (tonnes CO₂e)',
        barmode='stack',
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_regional, width = 'stretch')
    
    # Regional table
    regional_totals['Total'] = regional_totals['Total'].apply(lambda x: f"{x:,.0f}")
    regional_totals['Scope 1'] = regional_totals['Scope 1'].apply(lambda x: f"{x:,.0f}")
    regional_totals['Scope 2'] = regional_totals['Scope 2'].apply(lambda x: f"{x:,.0f}")
    regional_totals['Scope 3'] = regional_totals['Scope 3'].apply(lambda x: f"{x:,.0f}")
    
    st.dataframe(regional_totals, use_container_width=True)
    
    st.markdown("---")
    
    # Year-over-Year Comparison
    st.markdown("### 📊 Year-over-Year Comparison")
    
    yearly_totals = filtered_df.groupby('year').agg({
        'scope1_tonnes': 'sum',
        'scope2_market_tonnes': 'sum',
        'scope3_tonnes': 'sum',
        'total_emissions': 'sum'
    }).reset_index()
    
    yearly_totals['YoY Change'] = yearly_totals['total_emissions'].pct_change() * 100
    
    fig_yoy = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Annual Emissions', 'Year-over-Year Change %')
    )
    
    fig_yoy.add_trace(
        go.Bar(
            x=yearly_totals['year'],
            y=yearly_totals['total_emissions'],
            marker_color='#1f77b4',
            name='Total Emissions'
        ),
        row=1, col=1
    )
    
    fig_yoy.add_trace(
        go.Scatter(
            x=yearly_totals['year'],
            y=yearly_totals['YoY Change'],
            mode='lines+markers',
            marker=dict(size=10),
            line=dict(color='#ff7f0e', width=3),
            name='YoY Change %'
        ),
        row=1, col=2
    )
    
    fig_yoy.update_xaxes(title_text="Year", row=1, col=1)
    fig_yoy.update_xaxes(title_text="Year", row=1, col=2)
    fig_yoy.update_yaxes(title_text="Emissions (tonnes CO₂e)", row=1, col=1)
    fig_yoy.update_yaxes(title_text="Change (%)", row=1, col=2)
    
    fig_yoy.update_layout(height=400, showlegend=False, template='plotly_white')
    
    st.plotly_chart(fig_yoy, width = 'stretch')
    
    # YoY table
    yearly_display = yearly_totals.copy()
    yearly_display['total_emissions'] = yearly_display['total_emissions'].apply(lambda x: f"{x:,.0f}")
    yearly_display['YoY Change'] = yearly_display['YoY Change'].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A")
    yearly_display.columns = ['Year', 'Scope 1', 'Scope 2', 'Scope 3', 'Total Emissions', 'YoY Change']
    
    st.dataframe(yearly_display, use_container_width=True)
    
    st.markdown("---")
    
    # Key Insights
    st.markdown("### 💡 Key Insights")
    
    insights = []
    
    # Insight 1: Dominant scope
    if scope3_pct > 70:
        insights.append(f"🔴 **Scope 3 dominates** at {scope3_pct:.0f}% of total footprint - supply chain decarbonization is critical")
    
    # Insight 2: YoY trend
    if yoy_change > 5:
        insights.append(f"📈 **Emissions increased** {yoy_change:.1f}% YoY - reduction efforts need acceleration")
    elif yoy_change < -5:
        insights.append(f"📉 **Emissions decreased** {abs(yoy_change):.1f}% YoY - on positive trajectory")
    else:
        insights.append(f"➡️ **Emissions stable** ({yoy_change:+.1f}% YoY) - incremental progress")
    
    # Insight 3: Renewable energy
    if avg_renewable < 50:
        insights.append(f"⚠️ **Low renewable energy** at {avg_renewable:.0f}% - major opportunity for Scope 2 reduction")
    elif avg_renewable > 75:
        insights.append(f"✅ **High renewable energy** at {avg_renewable:.0f}% - strong progress on Scope 2")
    
    # Insight 4: Facility concentration
    top_3_pct = (facility_totals.head(3)['total_emissions'].sum() / total_emissions) * 100
    insights.append(f"🎯 **Top 3 facilities** account for {top_3_pct:.0f}% of emissions - targeted interventions recommended")
    
    for insight in insights:
        st.markdown(insight)
    
    # ============================================
    # NEW SECTION: INTENSITY METRICS
    # ============================================
    st.markdown("---")
    st.markdown("### 📉 Emissions Intensity Metrics")
    st.markdown("""
    **Why intensity matters:** Absolute emissions may grow with business expansion, but intensity 
    (emissions per unit of activity) shows operational efficiency improvements.
    """)
    
    # Calculate intensity metrics
    try:
        # Load operational metrics for intensity calculations
        from utils.data_loader import load_combined_metrics
        
        combined_data = load_combined_metrics()
        
        if len(combined_data) > 0:
            # Monthly aggregates
            monthly_intensity = combined_data.groupby(pd.Grouper(key='date', freq='M')).agg({
                'total_emissions': 'sum',
                'electricity_mwh': 'sum',
                'water_consumption_gallons': 'sum'
            }).reset_index()
            
            # Calculate intensities
            monthly_intensity['emissions_per_mwh'] = (
                monthly_intensity['total_emissions'] / monthly_intensity['electricity_mwh']
            ) * 1000  # kg CO2e per MWh
            
            monthly_intensity['emissions_per_mgal_water'] = (
                monthly_intensity['total_emissions'] / (monthly_intensity['water_consumption_gallons'] / 1_000_000)
            )  # tonnes per million gallons
            
            # Remove infinities and NaNs
            monthly_intensity = monthly_intensity.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(monthly_intensity) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Emissions per MWh trend
                    fig_intensity_energy = go.Figure()
                    
                    fig_intensity_energy.add_trace(go.Scatter(
                        x=monthly_intensity['date'],
                        y=monthly_intensity['emissions_per_mwh'],
                        mode='lines+markers',
                        name='Emissions Intensity',
                        line=dict(color='#3498db', width=3),
                        marker=dict(size=6)
                    ))
                    
                    # Add trendline
                    z = np.polyfit(range(len(monthly_intensity)), monthly_intensity['emissions_per_mwh'], 1)
                    p = np.poly1d(z)
                    
                    fig_intensity_energy.add_trace(go.Scatter(
                        x=monthly_intensity['date'],
                        y=p(range(len(monthly_intensity))),
                        mode='lines',
                        name='Trend',
                        line=dict(color='red', width=2, dash='dash')
                    ))
                    
                    fig_intensity_energy.update_layout(
                        title='Emissions Intensity per Energy Consumption',
                        xaxis_title='Date',
                        yaxis_title='kg CO₂e per MWh',
                        height=400,
                        template='plotly_white',
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_intensity_energy, width = 'stretch')
                    
                    # Calculate improvement
                    if len(monthly_intensity) >= 2:
                        first_value = monthly_intensity['emissions_per_mwh'].iloc[0]
                        last_value = monthly_intensity['emissions_per_mwh'].iloc[-1]
                        pct_change = ((last_value - first_value) / first_value) * 100
                        
                        if pct_change < 0:
                            st.success(f"✅ Emissions intensity improved {abs(pct_change):.1f}% over the period")
                        else:
                            st.warning(f"⚠️ Emissions intensity increased {pct_change:.1f}% over the period")
                
                with col2:
                    # Emissions per water consumed
                    fig_intensity_water = go.Figure()
                    
                    fig_intensity_water.add_trace(go.Scatter(
                        x=monthly_intensity['date'],
                        y=monthly_intensity['emissions_per_mgal_water'],
                        mode='lines+markers',
                        name='Water Intensity',
                        line=dict(color='#1abc9c', width=3),
                        marker=dict(size=6)
                    ))
                    
                    # Add trendline
                    z_water = np.polyfit(range(len(monthly_intensity)), monthly_intensity['emissions_per_mgal_water'], 1)
                    p_water = np.poly1d(z_water)
                    
                    fig_intensity_water.add_trace(go.Scatter(
                        x=monthly_intensity['date'],
                        y=p_water(range(len(monthly_intensity))),
                        mode='lines',
                        name='Trend',
                        line=dict(color='red', width=2, dash='dash')
                    ))
                    
                    fig_intensity_water.update_layout(
                        title='Emissions per Water Consumption',
                        xaxis_title='Date',
                        yaxis_title='Tonnes CO₂e per Million Gallons',
                        height=400,
                        template='plotly_white',
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_intensity_water, width = 'stretch')
                    
                    # Water efficiency insight
                    if len(monthly_intensity) >= 2:
                        first_water = monthly_intensity['emissions_per_mgal_water'].iloc[0]
                        last_water = monthly_intensity['emissions_per_mgal_water'].iloc[-1]
                        water_pct_change = ((last_water - first_water) / first_water) * 100
                        
                        if water_pct_change < 0:
                            st.success(f"✅ Water efficiency improved {abs(water_pct_change):.1f}% over the period")
                        else:
                            st.info(f"ℹ️ Water intensity changed {water_pct_change:+.1f}% (may reflect facility mix changes)")
                
                # Comparison table
                st.markdown("---")
                st.markdown("#### 📊 Absolute vs Intensity Metrics")
                
                # Compare first and last year
                if len(monthly_intensity) >= 12:
                    first_year = monthly_intensity.head(12).agg({
                        'total_emissions': 'sum',
                        'electricity_mwh': 'sum',
                        'emissions_per_mwh': 'mean'
                    })
                    
                    last_year = monthly_intensity.tail(12).agg({
                        'total_emissions': 'sum',
                        'electricity_mwh': 'sum',
                        'emissions_per_mwh': 'mean'
                    })
                    
                    comparison_df = pd.DataFrame({
                        'Metric': [
                            'Total Emissions (tonnes)',
                            'Total Electricity (MWh)',
                            'Emissions Intensity (kg/MWh)'
                        ],
                        'First Year': [
                            f"{first_year['total_emissions']:,.0f}",
                            f"{first_year['electricity_mwh']:,.0f}",
                            f"{first_year['emissions_per_mwh']:.1f}"
                        ],
                        'Latest Year': [
                            f"{last_year['total_emissions']:,.0f}",
                            f"{last_year['electricity_mwh']:,.0f}",
                            f"{last_year['emissions_per_mwh']:.1f}"
                        ],
                        'Change': [
                            f"{((last_year['total_emissions'] - first_year['total_emissions']) / first_year['total_emissions'] * 100):+.1f}%",
                            f"{((last_year['electricity_mwh'] - first_year['electricity_mwh']) / first_year['electricity_mwh'] * 100):+.1f}%",
                            f"{((last_year['emissions_per_mwh'] - first_year['emissions_per_mwh']) / first_year['emissions_per_mwh'] * 100):+.1f}%"
                        ]
                    })
                    
                    st.dataframe(comparison_df, width = use_container_width=True)
                    
                    st.info("""
                    **Key Insight:** While absolute emissions may increase with business growth (more data centers, 
                    more compute capacity), improving intensity metrics shows operational efficiency gains through:
                    - Renewable energy adoption (reduces grid emissions per MWh)
                    - PUE improvements (less energy per unit of IT load)
                    - Water efficiency (less cooling energy per gallon)
                    """)
            else:
                st.warning("Insufficient data for intensity calculations")
        else:
            st.info("Enable operational metrics to see intensity trends")
    
    except Exception as e:
        st.warning(f"Could not calculate intensity metrics: {e}")

    st.markdown("---")
    
    # Export
    st.markdown("### 💾 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Emissions Data (CSV)",
            data=csv,
            file_name=f"emissions_overview_{latest_year}.csv",
            mime="text/csv"
        )
    
    with col2:
        summary_data = {
            'Metric': ['Total Emissions', 'Scope 1', 'Scope 2', 'Scope 3', 'YoY Change', 'Avg Renewable %'],
            'Value': [
                f"{total_emissions:,.0f} tonnes",
                f"{total_scope1:,.0f} tonnes",
                f"{total_scope2:,.0f} tonnes",
                f"{total_scope3:,.0f} tonnes",
                f"{yoy_change:+.1f}%",
                f"{avg_renewable:.1f}%"
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_csv = summary_df.to_csv(index=False)
        
        st.download_button(
            label="📊 Download Summary (CSV)",
            data=summary_csv,
            file_name=f"emissions_summary_{latest_year}.csv",
            mime="text/csv"
        )

except Exception as e:
    st.error(f"Error loading data: {e}")
    import traceback
    st.code(traceback.format_exc())