import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
sys.path.append('..')
from utils.ml_models import FacilityClusterer
import sqlite3

st.set_page_config(page_title="Facility Clustering", page_icon="🏭", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #6f42c1;
        margin-bottom: 1rem;
    }
    .cluster-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .cluster-0 {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .cluster-1 {
        background-color: #f3e5f5;
        border-left: 5px solid #9c27b0;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .cluster-2 {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .cluster-3 {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🏭 AI-Powered Facility Clustering</p>', unsafe_allow_html=True)

st.markdown("""
## Unsupervised Learning for Facility Segmentation

Using K-means clustering to group facilities by emission patterns and characteristics:
- **Peer Group Identification**: Find similar facilities for benchmarking
- **Pattern Discovery**: Uncover hidden facility archetypes
- **Tailored Strategies**: Different reduction approaches for different clusters
- **Performance Tracking**: Monitor progress within peer groups

**Business Value:** Enable apples-to-apples comparisons, share best practices within clusters, customize initiatives.
""")

# Sidebar controls
st.sidebar.header("🎛️ Clustering Settings")

n_clusters = st.sidebar.slider(
    "Number of Clusters",
    min_value=2,
    max_value=8,
    value=4,
    help="How many facility groups to create"
)

clustering_features = st.sidebar.multiselect(
    "Clustering Features",
    ['Avg Scope 1+2', 'Avg Scope 3', 'Emission Intensity', 'Renewable Energy %', 'Facility Size', 'Region'],
    default=['Avg Scope 1+2', 'Avg Scope 3', 'Emission Intensity', 'Renewable Energy %'],
    help="Characteristics to use for clustering"
)

show_elbow = st.sidebar.checkbox(
    "Show Elbow Plot",
    value=True,
    help="Helps determine optimal number of clusters"
)

# Load data
@st.cache_data
def load_facility_profiles():
    """Load aggregated facility characteristics"""
    conn = sqlite3.connect('../data/sustainability_data.db')
    
    query = """
    SELECT 
        e.facility_id,
        f.facility_name,
        f.region,
        f.facility_type,
        AVG(e.scope1_tonnes + e.scope2_market_tonnes) as avg_scope12,
        AVG(e.scope3_tonnes) as avg_scope3,
        AVG(e.renewable_pct) as avg_renewable_pct,
        SUM(e.scope1_tonnes + e.scope2_market_tonnes + e.scope3_tonnes) as total_emissions,
        AVG(b.square_feet) as avg_square_feet,
        AVG(b.revenue_millions) as avg_revenue,
        SUM(e.scope1_tonnes + e.scope2_market_tonnes + e.scope3_tonnes) / NULLIF(SUM(b.revenue_millions), 0) as emission_intensity,
        COUNT(DISTINCT e.date) as months_of_data
    FROM emissions_monthly e
    JOIN facilities f ON e.facility_id = f.facility_id
    LEFT JOIN business_metrics b ON e.facility_id = b.facility_id AND e.date = b.date
    GROUP BY e.facility_id
    HAVING COUNT(DISTINCT e.date) >= 6
    ORDER BY total_emissions DESC
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df

try:
    data = load_facility_profiles()
    
    # Data summary
    st.subheader("📊 Facility Portfolio Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Facilities",
            data['facility_id'].nunique(),
            help="Facilities with sufficient data"
        )
    
    with col2:
        total_emissions = data['total_emissions'].sum()
        st.metric(
            "Total Emissions",
            f"{total_emissions/1000:,.0f}k tonnes",
            help="Cumulative emissions across all facilities"
        )
    
    with col3:
        avg_intensity = data['emission_intensity'].mean()
        st.metric(
            "Avg Intensity",
            f"{avg_intensity:,.0f} t/$M",
            help="Average emission intensity"
        )
    
    with col4:
        avg_renewable = data['avg_renewable_pct'].mean()
        st.metric(
            "Avg Renewable %",
            f"{avg_renewable:.0f}%",
            help="Average renewable energy percentage"
        )
    
    st.markdown("---")
    
    # Elbow plot
    if show_elbow:
        st.subheader("📈 Elbow Method - Optimal Cluster Selection")
        
        st.markdown("""
        The elbow plot helps identify the optimal number of clusters by showing where 
        adding more clusters provides diminishing returns.
        """)
        
        # Map feature selection
        feature_map = {
            'Avg Scope 1+2': 'avg_scope12',
            'Avg Scope 3': 'avg_scope3',
            'Emission Intensity': 'emission_intensity',
            'Renewable Energy %': 'avg_renewable_pct',
            'Facility Size': 'avg_square_feet',
            'Region': 'region'
        }
        
        cluster_feature_cols = [feature_map[f] for f in clustering_features if f != 'Region']
        
        # Handle region encoding if selected
        data_for_clustering = data.copy()
        if 'Region' in clustering_features:
            # One-hot encode region
            region_dummies = pd.get_dummies(data['region'], prefix='region')
            data_for_clustering = pd.concat([data_for_clustering, region_dummies], axis=1)
            cluster_feature_cols.extend(region_dummies.columns.tolist())
        
        with st.spinner("Calculating optimal clusters..."):
            clusterer_temp = FacilityClusterer(n_clusters=n_clusters)
            k_range, inertias = clusterer_temp.get_elbow_data(
                data_for_clustering,
                cluster_feature_cols,
                k_range=range(2, min(10, len(data)-1))
            )
        
        # Plot elbow
        fig_elbow = go.Figure()
        
        fig_elbow.add_trace(go.Scatter(
            x=k_range,
            y=inertias,
            mode='lines+markers',
            marker=dict(size=10, color='lightblue', line=dict(width=2, color='darkblue')),
            line=dict(width=3, color='darkblue'),
            name='Inertia'
        ))
        
        # Highlight selected k
        selected_idx = k_range.index(n_clusters) if n_clusters in k_range else None
        if selected_idx is not None:
            fig_elbow.add_trace(go.Scatter(
                x=[n_clusters],
                y=[inertias[selected_idx]],
                mode='markers',
                marker=dict(size=20, color='red', symbol='star'),
                name='Selected K',
                showlegend=True
            ))
        
        fig_elbow.update_layout(
            title='Elbow Plot - Within-Cluster Sum of Squares',
            xaxis_title='Number of Clusters (K)',
            yaxis_title='Inertia (Within-Cluster SS)',
            height=400,
            template='plotly_white',
            hovermode='x'
        )
        
        st.plotly_chart(fig_elbow, width = 'stretch')
        
        st.info("""
        💡 **How to read:** Look for the "elbow" where the curve starts to flatten. 
        This suggests the optimal number of clusters where additional clusters provide limited benefit.
        """)
        
        st.markdown("---")
    
    # Run clustering
    st.subheader("🤖 Clustering Analysis")
    
    with st.spinner(f"Running K-means clustering with {n_clusters} clusters..."):
        clusterer = FacilityClusterer(n_clusters=n_clusters)
        clustered_data = clusterer.fit_predict(data_for_clustering, cluster_feature_cols)
    
    st.success(f"✅ Clustered {len(data)} facilities into {n_clusters} groups!")
    
    # Cluster distribution
    cluster_counts = clustered_data['cluster'].value_counts().sort_index()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Pie chart
        fig_pie = go.Figure(data=[go.Pie(
            labels=[f'Cluster {i}' for i in cluster_counts.index],
            values=cluster_counts.values,
            hole=0.4,
            marker=dict(colors=px.colors.qualitative.Set3)
        )])
        
        fig_pie.update_layout(
            title='Facility Distribution by Cluster',
            height=350
        )
        
        st.plotly_chart(fig_pie, width = 'stretch')
    
    with col2:
        st.markdown("### 📊 Cluster Sizes")
        for cluster_id, count in cluster_counts.items():
            pct = count / len(data) * 100
            st.metric(
                f"Cluster {cluster_id}",
                f"{count} facilities",
                delta=f"{pct:.0f}%"
            )
    
    # PCA visualization
    st.subheader("🗺️ Cluster Visualization (PCA)")
    
    st.markdown("""
    Principal Component Analysis (PCA) reduces the multidimensional clustering features 
    to 2D for visualization. Proximity indicates similarity.
    """)
    
    variance = clusterer.get_pca_variance()
    
    fig_pca = go.Figure()
    
    # Plot each cluster
    colors = px.colors.qualitative.Set3
    
    for cluster_id in range(n_clusters):
        cluster_subset = clustered_data[clustered_data['cluster'] == cluster_id]
        
        fig_pca.add_trace(go.Scatter(
            x=cluster_subset['pca1'],
            y=cluster_subset['pca2'],
            mode='markers+text',
            name=f'Cluster {cluster_id}',
            text=cluster_subset['facility_name'],
            textposition='top center',
            textfont=dict(size=8),
            marker=dict(
                size=15,
                color=colors[cluster_id % len(colors)],
                line=dict(width=2, color='white'),
                opacity=0.8
            ),
            hovertemplate='<b>%{text}</b><br>PCA1: %{x:.2f}<br>PCA2: %{y:.2f}<extra></extra>'
        ))
    
    fig_pca.update_layout(
        title=f'Facility Clusters - PCA Projection',
        xaxis_title=f'PC1 ({variance[0]*100:.1f}% variance)' if variance is not None else 'PC1',
        yaxis_title=f'PC2 ({variance[1]*100:.1f}% variance)' if variance is not None else 'PC2',
        height=600,
        template='plotly_white',
        showlegend=True
    )
    
    st.plotly_chart(fig_pca, width = 'stretch')
    
    st.markdown("---")
    
    # Cluster profiles
    st.subheader("📋 Cluster Profiles & Characteristics")
    
    # Calculate cluster statistics
    cluster_profiles = clustered_data.groupby('cluster').agg({
        'avg_scope12': ['mean', 'std'],
        'avg_scope3': ['mean', 'std'],
        'emission_intensity': ['mean', 'std'],
        'avg_renewable_pct': ['mean', 'std'],
        'total_emissions': 'sum',
        'facility_id': 'count'
    }).round(0)
    
    cluster_profiles.columns = [
        'Scope 1+2 Mean', 'Scope 1+2 Std',
        'Scope 3 Mean', 'Scope 3 Std',
        'Intensity Mean', 'Intensity Std',
        'Renewable % Mean', 'Renewable % Std',
        'Total Emissions', 'Facility Count'
    ]
    
    st.dataframe(
        cluster_profiles.style
            .format({
                col: '{:,.0f}' for col in cluster_profiles.columns
            })
            .background_gradient(subset=['Total Emissions'], cmap='Reds')
            .background_gradient(subset=['Renewable % Mean'], cmap='Greens'),
        width = 'stretch'
    )
    
    # Detailed cluster analysis
    st.subheader("🔍 Detailed Cluster Analysis")
    
    selected_cluster = st.selectbox(
        "Select Cluster to Analyze",
        range(n_clusters),
        format_func=lambda x: f"Cluster {x} ({cluster_counts[x]} facilities)"
    )
    
    cluster_data = clustered_data[clustered_data['cluster'] == selected_cluster]
    
    # Cluster characteristics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_scope12 = cluster_data['avg_scope12'].mean()
        st.metric(
            "Avg Scope 1+2",
            f"{avg_scope12:,.0f} tonnes/mo",
            help="Average monthly operational emissions"
        )
    
    with col2:
        avg_intensity = cluster_data['emission_intensity'].mean()
        st.metric(
            "Avg Intensity",
            f"{avg_intensity:,.0f} t/$M",
            help="Emissions per million dollars revenue"
        )
    
    with col3:
        avg_renewable = cluster_data['avg_renewable_pct'].mean()
        st.metric(
            "Avg Renewable %",
            f"{avg_renewable:.0f}%",
            help="Average renewable energy percentage"
        )
    
    # Facilities in cluster
    st.markdown(f"#### Facilities in Cluster {selected_cluster}")
    
    cluster_facilities = cluster_data[[
        'facility_name', 'region', 'facility_type',
        'avg_scope12', 'avg_scope3', 'emission_intensity',
        'avg_renewable_pct', 'total_emissions'
    ]].copy()
    
    cluster_facilities.columns = [
        'Facility', 'Region', 'Type',
        'Avg Scope 1+2', 'Avg Scope 3', 'Intensity',
        'Renewable %', 'Total Emissions'
    ]
    
    st.dataframe(
        cluster_facilities.style
            .format({
                'Avg Scope 1+2': '{:,.0f}',
                'Avg Scope 3': '{:,.0f}',
                'Intensity': '{:,.0f}',
                'Renewable %': '{:.0f}',
                'Total Emissions': '{:,.0f}'
            })
            .background_gradient(subset=['Total Emissions'], cmap='Reds')
            .background_gradient(subset=['Renewable %'], cmap='Greens'),
        width = 'stretch'
    )
    
    # Cluster comparison radar chart
    st.markdown("#### 📊 Cluster Comparison - Radar Chart")
    
    # Normalize features for radar chart
    features_for_radar = ['avg_scope12', 'avg_scope3', 'emission_intensity', 'avg_renewable_pct']
    feature_labels = ['Scope 1+2', 'Scope 3', 'Intensity', 'Renewable %']
    
    normalized_data = clustered_data[features_for_radar].copy()
    for col in features_for_radar:
        if col == 'avg_renewable_pct':
            # Higher is better for renewable
            normalized_data[col] = normalized_data[col] / 100
        else:
            # Lower is better for emissions, so invert
            max_val = normalized_data[col].max()
            normalized_data[col] = 1 - (normalized_data[col] / max_val)
    
    cluster_means = normalized_data.groupby(clustered_data['cluster']).mean()
    
    fig_radar = go.Figure()
    
    for cluster_id in range(n_clusters):
        values = cluster_means.loc[cluster_id, features_for_radar].tolist()
        values.append(values[0])  # Close the polygon
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=feature_labels + [feature_labels[0]],
            fill='toself',
            name=f'Cluster {cluster_id}',
            line=dict(color=colors[cluster_id % len(colors)], width=2)
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title='Cluster Performance Comparison (Normalized)',
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig_radar, width = 'stretch')
    
    st.info("""
    💡 **Reading the chart:** Values closer to the edge are better. 
    For emissions/intensity, we've inverted the scale (lower = better). 
    For renewable %, higher values are at the edge.
    """)
    
    st.markdown("---")
    
    # Recommendations by cluster
    st.subheader("💡 Tailored Recommendations by Cluster")
    
    # Generate recommendations based on cluster characteristics
    recommendations = {}
    
    for cluster_id in range(n_clusters):
        cluster_subset = clustered_data[clustered_data['cluster'] == cluster_id]
        
        avg_scope12 = cluster_subset['avg_scope12'].mean()
        avg_scope3 = cluster_subset['avg_scope3'].mean()
        avg_intensity = cluster_subset['emission_intensity'].mean()
        avg_renewable = cluster_subset['avg_renewable_pct'].mean()
        
        # Classify cluster
        if avg_scope12 < clustered_data['avg_scope12'].median() and avg_renewable > 70:
            cluster_type = "High Performers"
            icon = "🏆"
            strategy = """
            **Strategy:** Share best practices
            - Document successful initiatives
            - Mentor other clusters
            - Push for even more aggressive targets
            - Focus on Scope 3 reduction
            """
        elif avg_intensity > clustered_data['emission_intensity'].quantile(0.75):
            cluster_type = "High Intensity - Efficiency Needed"
            icon = "🔧"
            strategy = """
            **Strategy:** Operational efficiency focus
            - Energy audits for all facilities
            - Upgrade equipment and processes
            - Implement ISO 50001 energy management
            - Quick wins: lighting, HVAC optimization
            """
        elif avg_renewable < 30:
            cluster_type = "Renewable Energy Opportunity"
            icon = "⚡"
            strategy = """
            **Strategy:** Accelerate renewable procurement
            - Evaluate PPA opportunities
            - Consider on-site solar/wind
            - Join renewable energy buyer consortium
            - Target 100% renewable by 2030
            """
        elif avg_scope3 > clustered_data['avg_scope3'].quantile(0.75):
            cluster_type = "Scope 3 Hotspot"
            icon = "🔗"
            strategy = """
            **Strategy:** Supply chain engagement
            - Map high-emission suppliers
            - Set supplier targets
            - Prefer low-carbon alternatives
            - Track Category 1 & 11 closely
            """
        else:
            cluster_type = "Solid Performers - Continuous Improvement"
            icon = "📈"
            strategy = """
            **Strategy:** Incremental improvements
            - Set 5% annual reduction targets
            - Benchmark against Cluster leaders
            - Invest in proven technologies
            - Monitor progress quarterly
            """
        
        recommendations[cluster_id] = {
            'type': cluster_type,
            'icon': icon,
            'strategy': strategy,
            'facilities': cluster_subset['facility_name'].tolist()
        }
    
    # Display recommendations
    for cluster_id, rec in recommendations.items():
        cluster_class = f"cluster-{cluster_id % 4}"
        
        facilities_list = ", ".join(rec['facilities'][:3])
        if len(rec['facilities']) > 3:
            facilities_list += f" (+{len(rec['facilities'])-3} more)"
        
        st.markdown(f"""
        <div class="{cluster_class}">
        <h4>{rec['icon']} Cluster {cluster_id}: {rec['type']}</h4>
        <p><strong>Facilities:</strong> {facilities_list}</p>
        {rec['strategy']}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Export functionality
    st.subheader("💾 Export Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export cluster assignments
        cluster_export = clustered_data[[
            'facility_id', 'facility_name', 'region', 'facility_type',
            'cluster', 'avg_scope12', 'avg_scope3', 'emission_intensity',
            'avg_renewable_pct', 'total_emissions'
        ]].copy()
        
        cluster_export.columns = [
            'Facility ID', 'Facility Name', 'Region', 'Type',
            'Cluster', 'Avg Scope 1+2', 'Avg Scope 3', 'Intensity',
            'Renewable %', 'Total Emissions'
        ]
        
        csv = cluster_export.to_csv(index=False)
        st.download_button(
            label="📥 Download Cluster Assignments (CSV)",
            data=csv,
            file_name=f"facility_clusters_{n_clusters}_clusters.csv",
            mime="text/csv"
        )
    
    with col2:
        # Export summary report
        report = f"""
FACILITY CLUSTERING ANALYSIS REPORT
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

CLUSTERING PARAMETERS:
- Number of Clusters: {n_clusters}
- Features Used: {', '.join(clustering_features)}
- Total Facilities: {len(data)}

CLUSTER DISTRIBUTION:
{chr(10).join([f"Cluster {cid}: {count} facilities ({count/len(data)*100:.0f}%)" 
               for cid, count in cluster_counts.items()])}

CLUSTER PROFILES:
{chr(10).join([f"""
Cluster {cid}: {recommendations[cid]['type']}
- Avg Scope 1+2: {cluster_profiles.loc[cid, 'Scope 1+2 Mean']:,.0f} tonnes/mo
- Avg Intensity: {cluster_profiles.loc[cid, 'Intensity Mean']:,.0f} t/$M
- Renewable %: {cluster_profiles.loc[cid, 'Renewable % Mean']:.0f}%
- Strategy: {recommendations[cid]['type']}
""" for cid in range(n_clusters)])}

KEY RECOMMENDATIONS:
1. Share best practices from high-performing clusters
2. Provide targeted support to high-intensity clusters
3. Set cluster-specific reduction targets
4. Monitor progress within peer groups quarterly

NEXT STEPS:
- Communicate cluster assignments to facility managers
- Schedule cluster-specific workshops
- Establish cluster benchmark dashboards
- Re-cluster annually to track movement
        """
        
        st.download_button(
            label="📄 Download Report (TXT)",
            data=report,
            file_name=f"clustering_report_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )

except Exception as e:
    st.error(f"Error loading data or performing clustering: {e}")
    st.info("Please ensure the database exists and contains facility-level emissions data.")
    import traceback
    st.code(traceback.format_exc())