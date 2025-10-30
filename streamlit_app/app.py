import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sqlite3
import sys
sys.path.append('.')
from utils.data_loader import get_data_quality_status, get_summary_statistics

st.set_page_config(
    page_title="Sustainability Analytics Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🌍 Corporate Sustainability Analytics Platform</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Enterprise Carbon Accounting & ESG Reporting Dashboard</p>', unsafe_allow_html=True)

# Data Quality Status
data_status = get_data_quality_status()

if data_status['using_cleaned']:
    st.info("ℹ️ **Data Status:** Using cleaned database (quality-validated data)")
else:
    st.info("""
    ℹ️ **Data Note:** This dataset is based on actual tech company environmental reports (2020-2024). 
    Emissions data matches published totals. Operational metrics include real facility-level water consumption 
    and PUE ratings where available.
    """)

st.markdown("---")

# Project Overview
st.markdown("## 📋 Project Overview")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### Enterprise-Grade Sustainability Analytics
    
    This dashboard demonstrates a **complete end-to-end analytics workflow** for corporate carbon accounting 
    based on real environmental report data:
    
    **📊 Carbon Accounting & Reporting**
    - GHG Protocol-aligned Scope 1, 2, 3 tracking
    - Data based on actual tech company reports (2020-2024)
    - Multi-facility, multi-region monitoring
    - Science-Based Target tracking and progress
    
    **🤖 Machine Learning & AI**
    - Time series forecasting (Prophet, Holt-Winters)
    - Anomaly detection (Isolation Forest, Z-score)
    - Emissions driver analysis (Random Forest, SHAP)
    - Facility clustering and segmentation (K-means, PCA)
    
    **📈 Operational Metrics & Efficiency**
    - Real facility-level water consumption data
    - PUE (Power Usage Effectiveness) tracking
    - Carbon-free energy (CFE) hourly matching
    - Waste circularity and diversion rates
    - ROI analysis for efficiency investments
    
    **💰 Financial Analysis**
    - Cost modeling (energy, water, carbon pricing)
    - ROI of decarbonization strategies
    - Efficiency investment payback analysis
    - Carbon pricing scenario modeling
    """)

with col2:
    st.markdown("""
    ### 🛠️ Technical Stack
    
    **Data & Engineering:**
    - Python, Pandas, NumPy
    - SQLite, SQL queries
    - Real environmental data
    - Operational metrics
    
    **Machine Learning:**
    - Scikit-learn
    - Facebook Prophet
    - Statsmodels
    - SHAP (explainability)
    
    **Visualization:**
    - Streamlit
    - Plotly (interactive)
    - Custom dashboards
    
    **Data Sources:**
    - Environmental reports (2020-2024)
    - EPA eGRID emission factors
    - Regional grid CFE data
    - Facility-level metrics
    """)

st.markdown("---")

# Load summary stats
try:
    stats = get_summary_statistics()
    
    st.markdown("## 📊 Current Performance Snapshot")
    
    # Top-level KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total = stats['scope_totals']['total']
        st.metric(
            "Total Emissions",
            f"{total/1_000_000:.1f}M tonnes",
            help="Cumulative emissions across all facilities and time periods"
        )
    
    with col2:
        recent = stats['recent_month']['total_emissions']
        st.metric(
            "Latest Month",
            f"{recent/1000:.0f}K tonnes",
            help="Most recent monthly emissions"
        )
    
    with col3:
        if 'yoy_change' in stats:
            yoy = stats['yoy_change']['pct_change']
            st.metric(
                "YoY Change",
                f"{yoy:+.1f}%",
                delta=f"{yoy:.1f}%",
                delta_color="inverse",
                help="Year-over-year change in emissions"
            )
        else:
            st.metric("YoY Change", "N/A")
    
    with col4:
        renewable_pct = 100.0  # 100% renewable electricity match
        st.metric(
            "Renewable Energy",
            f"{renewable_pct:.0f}%",
            help="Percentage of electricity matched with renewable sources annually"
        )
    
    with col5:
        scope1_pct = (stats['scope_totals']['scope1'] / stats['scope_totals']['total']) * 100
        st.metric(
            "Scope 1",
            f"{scope1_pct:.1f}%",
            help="Direct emissions as percentage of total"
        )
    
    st.markdown("---")
    
    # Operational Excellence Metrics
    st.markdown("### ⚡ Operational Excellence Metrics")
    st.markdown("*Key performance indicators based on actual facility data*")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_pue = stats['operational']['avg_pue']
        if avg_pue:
            st.metric(
                "Fleet PUE",
                f"{avg_pue:.2f}",
                help="Power Usage Effectiveness - Industry leading efficiency (Google: 1.09)"
            )
        else:
            st.metric("Fleet PUE", "N/A")
    
    with col2:
        avg_cfe = stats['operational']['avg_cfe']
        if avg_cfe:
            st.metric(
                "Carbon-Free Energy",
                f"{avg_cfe*100:.0f}%",
                help="Hourly match of carbon-free energy (harder than 100% annual match)"
            )
        else:
            st.metric("Carbon-Free Energy", "N/A")
    
    with col3:
        avg_water_replen = stats['operational']['avg_water_replen']
        if avg_water_replen:
            st.metric(
                "Water Replenishment",
                f"{avg_water_replen*100:.0f}%",
                help="Percentage of water consumption replenished (Goal: 100% by 2030)"
            )
        else:
            st.metric("Water Replenishment", "N/A")
    
    with col4:
        avg_waste_div = stats['operational']['avg_waste_diversion']
        if avg_waste_div:
            st.metric(
                "Waste Diversion",
                f"{avg_waste_div*100:.0f}%",
                help="Percentage of waste diverted from landfills"
            )
        else:
            st.metric("Waste Diversion", "N/A")
    
    st.markdown("---")
    
    # Emissions breakdown visualization
    st.markdown("### 📈 Emissions by Scope")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        scope_data = pd.DataFrame({
            'Scope': ['Scope 1\n(Direct)', 'Scope 2\n(Electricity)', 'Scope 3\n(Value Chain)'],
            'Emissions': [
                stats['scope_totals']['scope1'],
                stats['scope_totals']['scope2'],
                stats['scope_totals']['scope3']
            ],
            'Color': ['#ff7f0e', '#2ca02c', '#d62728']
        })
        
        fig_scope = go.Figure(data=[
            go.Bar(
                x=scope_data['Scope'],
                y=scope_data['Emissions'],
                marker_color=scope_data['Color'],
                text=scope_data['Emissions'].apply(lambda x: f"{x/1_000_000:.1f}M"),
                textposition='outside'
            )
        ])
        
        fig_scope.update_layout(
            yaxis_title='Emissions (tonnes CO₂e)',
            height=400,
            template='plotly_white',
            showlegend=False
        )
        
        st.plotly_chart(fig_scope, width = 'stretch'
    
    with col2:
        # Scope breakdown percentages
        total_emissions = stats['scope_totals']['total']
        
        scope_pcts = pd.DataFrame({
            'Scope': ['Scope 1', 'Scope 2', 'Scope 3'],
            'Percentage': [
                (stats['scope_totals']['scope1'] / total_emissions) * 100,
                (stats['scope_totals']['scope2'] / total_emissions) * 100,
                (stats['scope_totals']['scope3'] / total_emissions) * 100
            ]
        })
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=scope_pcts['Scope'],
            values=scope_pcts['Percentage'],
            marker_colors=['#ff7f0e', '#2ca02c', '#d62728'],
            hole=0.4,
            textinfo='label+percent',
            textposition='outside'
        )])
        
        fig_pie.update_layout(
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_pie, width = 'stretch'

except Exception as e:
    st.warning(f"Unable to load summary statistics: {e}")
    st.info("Please ensure the database has been generated using `python data/generate_sample_data.py`")

st.markdown("---")

# Key Features
st.markdown("## ✨ Dashboard Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-box">
    <h4>📊 Emissions Overview</h4>
    <ul>
        <li>Real-time KPI tracking</li>
        <li>Multi-dimensional analysis</li>
        <li>Trend visualization</li>
        <li>Regional comparisons</li>
        <li>Intensity metrics</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-box">
    <h4>🔬 Scope Analysis</h4>
    <ul>
        <li>GHG Protocol breakdown</li>
        <li>Scope 3 category analysis</li>
        <li>Waterfall charts</li>
        <li>Pareto analysis</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-box">
    <h4>🤖 AI Forecasting</h4>
    <ul>
        <li>Prophet time series</li>
        <li>Holt-Winters smoothing</li>
        <li>Confidence intervals</li>
        <li>Trend decomposition</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-box">
    <h4>🚨 Anomaly Detection</h4>
    <ul>
        <li>Isolation Forest (ML)</li>
        <li>Statistical Z-score</li>
        <li>Investigation workflow</li>
        <li>Root cause analysis</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-box">
    <h4>📈 Driver Analysis</h4>
    <ul>
        <li>Operational metrics</li>
        <li>Feature importance</li>
        <li>SHAP explainability</li>
        <li>What-if scenarios</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-box">
    <h4>🏭 Facility Clustering</h4>
    <ul>
        <li>K-means segmentation</li>
        <li>PCA visualization</li>
        <li>Peer benchmarking</li>
        <li>Targeted strategies</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-box">
    <h4>💾 Data Quality</h4>
    <ul>
        <li>5-dimensional validation</li>
        <li>Completeness checks</li>
        <li>Consistency rules</li>
        <li>Quality scoring</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-box">
    <h4>💰 ROI Analysis</h4>
    <ul>
        <li>Efficiency investments</li>
        <li>Renewable energy ROI</li>
        <li>Carbon pricing impact</li>
        <li>Decarbonization costs</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Skills Demonstrated
st.markdown("## 🎯 Skills Demonstrated")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Data Engineering & Analytics
    - ✅ **Real Data Integration**: Based on actual environmental reports (2020-2024)
    - ✅ **SQL & Databases**: Complex queries, joins, aggregations, schema design
    - ✅ **Python Data Stack**: Pandas, NumPy, efficient data manipulation
    - ✅ **Data Modeling**: Operational metrics, emissions tracking, cost modeling
    - ✅ **Statistical Analysis**: Trends, correlations, hypothesis testing
    
    ### Machine Learning
    - ✅ **Time Series**: Prophet, ARIMA, Holt-Winters forecasting
    - ✅ **Anomaly Detection**: Isolation Forest, statistical outlier methods
    - ✅ **Supervised Learning**: Regression, classification, ensemble models
    - ✅ **Unsupervised Learning**: K-means, PCA, hierarchical clustering
    - ✅ **Model Explainability**: SHAP values, feature importance analysis
    """)

with col2:
    st.markdown("""
    ### Domain Expertise
    - ✅ **Sustainability Metrics**: Emissions, PUE, CFE, water intensity
    - ✅ **GHG Protocol**: Scope 1, 2, 3 accounting standards
    - ✅ **Industry Benchmarks**: Tech sector sustainability performance
    - ✅ **Financial Analysis**: ROI, cost modeling, carbon pricing
    
    ### Software Engineering
    - ✅ **Modular Architecture**: Reusable components, clean code structure
    - ✅ **Interactive Dashboards**: Streamlit, Plotly visualizations
    - ✅ **Documentation**: Comprehensive inline and external docs
    - ✅ **Best Practices**: Error handling, logging, maintainability
    - ✅ **Production Ready**: Scalable, enterprise-grade patterns
    """)

st.markdown("---")

# Data Sources
st.markdown("## 📚 Data Sources & Methodology")

st.markdown("""
### Real Environmental Data

This dashboard uses actual data from major tech company environmental reports:

**Emissions Data (2020-2024):**
- Annual totals by scope (Scope 1, 2, 3)
- 100% renewable electricity matching
- Growth from 8.7M to 15.2M tonnes CO₂e

**Operational Metrics:**
- Facility-level water consumption (27 data centers with actual 2024 data)
- PUE ratings (27 facilities, 2020-2024 trends)
- Regional carbon-free energy percentages
- Waste generation and diversion rates

**Disaggregation Methodology:**
- Annual totals → Monthly allocation with seasonal patterns
- Facility-level data where available from reports
- Proportional allocation based on facility type for estimates
- Regional grid factors from EPA eGRID and EEA

**Why This Approach:**
- Companies publish annual aggregates, not monthly facility-level data
- Demonstrates ability to work with real-world data constraints
- Shows understanding of sustainability reporting standards
- Patterns match actual industry trends (AI boom, renewable adoption)
""")

st.markdown("---")

# Getting Started
st.markdown("## 🚀 Getting Started")

st.markdown("""
### Recommended Navigation Path:

1. **📊 Emissions Overview** - Review emissions landscape and trends
2. **🔬 Scope Analysis** - Deep dive into GHG Protocol breakdown
3. **⚡ Operational Metrics** - Explore efficiency and resource use
4. **🤖 AI Forecasting** - Predictive models and scenario planning
5. **📈 Driver Analysis** - Understand what drives emissions
6. **💰 ROI Analysis** - Financial impact of decarbonization strategies
7. **🏭 Facility Clustering** - Benchmark and segment facilities
8. **🚨 Anomaly Detection** - Investigate unusual patterns
9. **💾 Data Quality** - Validation and quality assurance

---

### 💼 Real-World Applications

**For Sustainability Teams:**
- Track progress toward Science-Based Targets
- Identify decarbonization opportunities
- Generate stakeholder reports (CDP, TCFD)

**For Data Teams:**
- Operational efficiency monitoring
- Predictive analytics for planning
- ROI analysis for investments

**For Finance Teams:**
- Carbon pricing scenario modeling
- Investment prioritization
- Cost-benefit analysis
""")

st.markdown("---")

st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🌍 Corporate Sustainability Analytics Platform</p>
    <p>Demonstrating Enterprise-Grade Data Engineering, ML & Analytics</p>
    <p><em>Based on real environmental report data with operational insights</em></p>
</div>
""", unsafe_allow_html=True)