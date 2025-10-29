import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
sys.path.append('..')
from utils.data_loader import load_emissions_data, load_facilities
import sqlite3

st.set_page_config(page_title="Scope Analysis", page_icon="🔬", layout="wide")

st.markdown('<style>.main-header {font-size: 2.5rem; font-weight: 700; color: #2ca02c;}</style>', unsafe_allow_html=True)
st.markdown('<p class="main-header">🔬 Scope Analysis</p>', unsafe_allow_html=True)

st.markdown("""
## GHG Protocol Scope Breakdown

Deep dive into Scope 1, 2, and 3 emissions following GHG Protocol methodology.

**GHG Protocol Scopes:**
- **Scope 1:** Direct emissions from owned/controlled sources
- **Scope 2:** Indirect emissions from purchased electricity
- **Scope 3:** All other indirect emissions in value chain
""")

@st.cache_data
def get_scope_data():
    emissions_df = load_emissions_data()
    facilities_df = load_facilities()
    emissions_df['date'] = pd.to_datetime(emissions_df['date'])
    emissions_df['year'] = emissions_df['date'].dt.year
    emissions_df['month'] = emissions_df['date'].dt.to_period('M').astype(str)
    conn = sqlite3.connect('../data/sustainability_data.db')
    scope3_categories = pd.read_sql_query("SELECT category_id, category_name, description, typical_pct_of_scope3 FROM scope3_categories ORDER BY typical_pct_of_scope3 DESC", conn)
    conn.close()
    return emissions_df, facilities_df, scope3_categories

try:
    emissions_df, facilities_df, scope3_categories = get_scope_data()
    
    st.sidebar.header("🎛️ Analysis Options")
    selected_year = st.sidebar.selectbox("Select Year", sorted(emissions_df['year'].unique(), reverse=True))
    scope_focus = st.sidebar.radio("Focus Area", ["All Scopes", "Scope 1", "Scope 2", "Scope 3"])
    
    year_data = emissions_df[emissions_df['year'] == selected_year]
    total_scope1 = year_data['scope1_tonnes'].sum()
    total_scope2_location = year_data['scope2_location_tonnes'].sum()
    total_scope2_market = year_data['scope2_market_tonnes'].sum()
    total_scope3 = year_data['scope3_tonnes'].sum()
    total_all = total_scope1 + total_scope2_market + total_scope3
    
    st.markdown(f"### 📊 {selected_year} Scope Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Scope 1 (Direct)", f"{total_scope1/1000:.1f}K tonnes", f"{(total_scope1/total_all)*100:.1f}%")
    with col2:
        st.metric("Scope 2 (Market)", f"{total_scope2_market/1000:.1f}K tonnes", f"{(total_scope2_market/total_all)*100:.1f}%")
    with col3:
        st.metric("Scope 3 (Value Chain)", f"{total_scope3/1000:.1f}K tonnes", f"{(total_scope3/total_all)*100:.1f}%")
    with col4:
        st.metric("Total Footprint", f"{total_all/1_000_000:.2f}M tonnes")
    
    st.markdown("---")
    
    if scope_focus in ["All Scopes", "Scope 1"]:
        st.markdown("### 🔥 Scope 1: Direct Emissions")
        st.info("**Sources:** Backup generators, natural gas heating, fleet vehicles, refrigerants")
        
        col1, col2 = st.columns(2)
        with col1:
            scope1_monthly = year_data.groupby('month')['scope1_tonnes'].sum().reset_index().sort_values('month')
            fig_s1 = go.Figure(go.Scatter(x=scope1_monthly['month'], y=scope1_monthly['scope1_tonnes'], mode='lines+markers', line=dict(color='#ff7f0e', width=3), fill='tozeroy'))
            fig_s1.update_layout(title='Scope 1 Monthly Trend', xaxis_title='Month', yaxis_title='Emissions (tonnes CO₂e)', height=400, template='plotly_white')
            st.plotly_chart(fig_s1, width = 'stretch')
        
        with col2:
            scope1_fac = year_data.groupby('facility_id')['scope1_tonnes'].sum().reset_index()
            scope1_fac = scope1_fac.merge(facilities_df[['facility_id', 'facility_name']], on='facility_id').sort_values('scope1_tonnes', ascending=False).head(10)
            fig_s1_fac = go.Figure(go.Bar(x=scope1_fac['scope1_tonnes'], y=scope1_fac['facility_name'], orientation='h', marker_color='#ff7f0e'))
            fig_s1_fac.update_layout(title='Top 10 Facilities - Scope 1', xaxis_title='Emissions', height=400, template='plotly_white', yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_s1_fac, width = 'stretch')
        st.markdown("---")
    
    if scope_focus in ["All Scopes", "Scope 2"]:
        st.markdown("### ⚡ Scope 2: Indirect Emissions (Electricity)")
        st.info("**Accounting Methods:** Location-based (grid average) vs Market-based (PPAs/RECs)")
        
        col1, col2 = st.columns(2)
        with col1:
            scope2_monthly = year_data.groupby('month').agg({'scope2_location_tonnes':'sum','scope2_market_tonnes':'sum'}).reset_index().sort_values('month')
            fig_s2 = go.Figure()
            fig_s2.add_trace(go.Scatter(x=scope2_monthly['month'], y=scope2_monthly['scope2_location_tonnes'], mode='lines', name='Location-Based', line=dict(color='#2ca02c', dash='dash')))
            fig_s2.add_trace(go.Scatter(x=scope2_monthly['month'], y=scope2_monthly['scope2_market_tonnes'], mode='lines+markers', name='Market-Based', line=dict(color='#2ca02c', width=3)))
            fig_s2.update_layout(title='Scope 2: Location vs Market', xaxis_title='Month', yaxis_title='Emissions', height=400, template='plotly_white')
            st.plotly_chart(fig_s2, width = 'stretch')
            
            renewable_impact = total_scope2_location - total_scope2_market
            renewable_pct = (renewable_impact / total_scope2_location) * 100
            st.success(f"**Renewable Impact:** {renewable_impact:,.0f} tonnes avoided ({renewable_pct:.1f}% reduction)")
        
        with col2:
            renewable_monthly = year_data.groupby('month')['renewable_pct'].mean().reset_index().sort_values('month')
            fig_ren = go.Figure(go.Scatter(x=renewable_monthly['month'], y=renewable_monthly['renewable_pct'], mode='lines+markers', line=dict(color='#2ca02c', width=3), fill='tozeroy'))
            fig_ren.update_layout(title='Average Renewable Energy %', xaxis_title='Month', yaxis_title='Renewable %', yaxis_range=[0,100], height=400, template='plotly_white')
            st.plotly_chart(fig_ren, width = 'stretch')
        st.markdown("---")
    
    if scope_focus in ["All Scopes", "Scope 3"]:
        st.markdown("### 🔗 Scope 3: Value Chain Emissions")
        st.info("**Top Categories:** Purchased goods (45%), Use of sold products (25%), Capital goods (15%)")
        
        scope3_breakdown = scope3_categories.copy()
        scope3_breakdown['estimated_tonnes'] = total_scope3 * scope3_breakdown['typical_pct_of_scope3'] / 100
        scope3_breakdown['cumulative_pct'] = scope3_breakdown['typical_pct_of_scope3'].cumsum()
        
        col1, col2 = st.columns(2)
        with col1:
            fig_scope3_bar = go.Figure(go.Bar(
                x = scope3_breakdown['category_name'],
                y = scope3_breakdown['estimated_tonnes'],
                marker = dict(
                    color = scope3_breakdown['estimated_tonnes'],
                    colorscale = 'Reds',
                    showscale = False
                ),
                text = scope3_breakdown['estimated_tonnes'].apply(lambda x: f"{x:,.0f}"),
                textposition = 'outside'
            ))

            fig_scope3_bar.update_layout(
                title = 'Scope 3 Category Breakdown',
                xaxis_title = 'Category',
                yaxis_title = 'Emissions (tonnes CO₂e)',
                height = 450,
                template = 'plotly_white',
                xaxis = {'tickangle': -45}
            )

            st.plotly_chart(fig_scope3_bar, width = 'stretch')
        
        with col2:
            fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
            fig_pareto.add_trace(go.Bar(x=scope3_breakdown['category_name'], y=scope3_breakdown['estimated_tonnes'], name='Emissions', marker_color='#d62728'), secondary_y=False)
            fig_pareto.add_trace(go.Scatter(x=scope3_breakdown['category_name'], y=scope3_breakdown['cumulative_pct'], name='Cumulative %', mode='lines+markers', marker=dict(color='#ff7f0e', size=8), line=dict(color='#ff7f0e', width=3)), secondary_y=True)
            fig_pareto.update_xaxes(title_text="Category", tickangle=-45)
            fig_pareto.update_yaxes(title_text="Emissions", secondary_y=False)
            fig_pareto.update_yaxes(title_text="Cumulative %", range=[0,100], secondary_y=True)
            fig_pareto.update_layout(title='Pareto Analysis', height=450, template='plotly_white')
            st.plotly_chart(fig_pareto, width = 'stretch')
        
        st.markdown("#### 📋 Scope 3 Categories")
        scope3_table = scope3_breakdown[['category_name','description','estimated_tonnes','typical_pct_of_scope3']].copy()
        scope3_table.columns = ['Category','Description','Estimated Emissions','% of Scope 3']
        scope3_table['Estimated Emissions'] = scope3_table['Estimated Emissions'].apply(lambda x: f"{x:,.0f}")
        scope3_table['% of Scope 3'] = scope3_table['% of Scope 3'].apply(lambda x: f"{x:.1f}%")
        st.dataframe(scope3_table, width = 'stretch')
        
        top_3_pct = scope3_breakdown.head(3)['typical_pct_of_scope3'].sum()
        st.success(f"**80/20 Rule:** Top 3 categories = {top_3_pct:.0f}% of Scope 3")
        st.markdown("---")
    
    st.markdown("### 📊 All Scopes - Stacked Area")
    monthly_all = year_data.groupby('month').agg({'scope1_tonnes':'sum','scope2_market_tonnes':'sum','scope3_tonnes':'sum'}).reset_index().sort_values('month')
    fig_all = go.Figure()
    fig_all.add_trace(go.Scatter(x=monthly_all['month'], y=monthly_all['scope1_tonnes'], name='Scope 1', mode='lines', stackgroup='one', fillcolor='rgba(255,127,14,0.6)'))
    fig_all.add_trace(go.Scatter(x=monthly_all['month'], y=monthly_all['scope2_market_tonnes'], name='Scope 2', mode='lines', stackgroup='one', fillcolor='rgba(44,160,44,0.6)'))
    fig_all.add_trace(go.Scatter(x=monthly_all['month'], y=monthly_all['scope3_tonnes'], name='Scope 3', mode='lines', stackgroup='one', fillcolor='rgba(214,39,40,0.6)'))
    fig_all.update_layout(title='Stacked Area - All Scopes', xaxis_title='Month', yaxis_title='Emissions', height=500, template='plotly_white', hovermode='x unified')
    st.plotly_chart(fig_all, width = 'stretch')
    
    st.markdown("---")
    st.markdown("### 🎯 Reduction Priorities")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Scope 1:** Electrify heating, EV fleet, renewable diesel - 20% reduction possible")
    with col2:
        st.markdown("**Scope 2:** Increase PPAs, energy efficiency, 24/7 CFE - 50% reduction possible")
    with col3:
        st.markdown("**Scope 3:** Supplier engagement, circular economy, SBTs - 30% reduction possible")
    
    csv = monthly_all.to_csv(index=False)
    st.download_button("📥 Download Scope Data", csv, f"scope_analysis_{selected_year}.csv", "text/csv")

except Exception as e:
    st.error(f"Error: {e}")
    import traceback
    st.code(traceback.format_exc())