import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import sys
sys.path.append('..')
from utils.data_loader import load_combined_metrics, load_facilities

st.set_page_config(page_title="Facility Clustering", page_icon="🏭", layout="wide")

st.markdown('<style>.main-header {font-size: 2.5rem; font-weight: 700; color: #e67e22;}</style>', unsafe_allow_html=True)
st.markdown('<p class="main-header">🏭 Facility Clustering & Segmentation</p>', unsafe_allow_html=True)

st.markdown("""
## Strategic Facility Segmentation

Use unsupervised machine learning to identify facility peer groups and develop 
cluster-specific decarbonization strategies.

**Analysis Approach:**
- **K-Means Clustering**: Group facilities by emissions and operational characteristics
- **PCA Visualization**: Reduce dimensionality for 2D/3D visualization
- **Cluster Profiles**: Identify characteristics and tailored strategies for each group
- **ROI Analysis**: Estimate cost-benefit of interventions by cluster type
""")

st.markdown("---")

# Load data
@st.cache_data
def load_data():
    """Load combined metrics and calculate intensities"""
    df = load_combined_metrics()
    facilities = load_facilities()
    
    # Calculate intensities and metrics
    df['emissions_per_mwh'] = (df['total_emissions'] / df['electricity_mwh']) * 1000  # kg/MWh
    df['water_per_mwh'] = df['water_consumption_gallons'] / df['electricity_mwh']  # gal/MWh
    
    # Aggregate by facility (average over time)
    facility_agg = df.groupby('facility_id').agg({
        'facility_name': 'first',
        'region': 'first',
        'facility_type': 'first',
        'total_emissions': 'mean',
        'electricity_mwh': 'mean',
        'pue': 'mean',
        'cfe_pct': 'mean',
        'water_consumption_gallons': 'mean',
        'waste_diversion_pct': 'mean',
        'emissions_per_mwh': 'mean',
        'water_per_mwh': 'mean',
        'energy_cost_usd': 'mean',
        'carbon_cost_usd': 'mean'
    }).reset_index()
    
    # Remove rows with missing critical data
    facility_agg = facility_agg.dropna(subset=['pue', 'cfe_pct', 'emissions_per_mwh'])
    
    return facility_agg

try:
    data = load_data()
    
    if len(data) < 5:
        st.error("Need at least 5 facilities for meaningful clustering")
        st.stop()
    
    st.success(f"✅ Loaded {len(data)} facilities for clustering analysis")
    
    # Sidebar Configuration
    st.sidebar.header("🎛️ Clustering Configuration")
    
    # Feature selection
    st.sidebar.subheader("📊 Features for Clustering")
    
    available_features = {
        'Total Emissions': 'total_emissions',
        'Emissions Intensity (kg/MWh)': 'emissions_per_mwh',
        'PUE (Efficiency)': 'pue',
        'Carbon-Free Energy %': 'cfe_pct',
        'Water Intensity (gal/MWh)': 'water_per_mwh',
        'Waste Diversion %': 'waste_diversion_pct',
        'Electricity Consumption': 'electricity_mwh'
    }
    
    selected_features = st.sidebar.multiselect(
        "Select Clustering Features",
        options=list(available_features.keys()),
        default=['Emissions Intensity (kg/MWh)', 'PUE (Efficiency)', 'Carbon-Free Energy %', 'Water Intensity (gal/MWh)'],
        help="Choose which metrics to use for grouping facilities"
    )
    
    if len(selected_features) < 2:
        st.warning("⚠️ Please select at least 2 features for clustering")
        st.stop()
    
    feature_cols = [available_features[f] for f in selected_features]
    
    # Number of clusters
    n_clusters = st.sidebar.slider(
        "Number of Clusters",
        min_value=2,
        max_value=min(8, len(data)-1),
        value=4,
        help="How many distinct facility groups to identify"
    )
    
    # Filter by facility type
    facility_types = st.sidebar.multiselect(
        "Filter by Facility Type",
        options=data['facility_type'].unique(),
        default=data['facility_type'].unique(),
        help="Focus analysis on specific facility types"
    )
    
    data_filtered = data[data['facility_type'].isin(facility_types)].copy()
    
    if len(data_filtered) < 5:
        st.error("Not enough facilities after filtering")
        st.stop()
    
    st.markdown("---")
    
    # Prepare features for clustering
    X = data_filtered[feature_cols].copy()
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering
    st.markdown("### 🤖 Clustering Analysis")
    
    with st.spinner("Performing clustering..."):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        data_filtered['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Calculate silhouette score
        silhouette = silhouette_score(X_scaled, data_filtered['cluster'])
    
    st.success(f"✅ Clustering complete! Silhouette Score: {silhouette:.3f}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Facilities Analyzed", len(data_filtered))
    
    with col2:
        st.metric("Clusters Identified", n_clusters)
    
    with col3:
        if silhouette > 0.5:
            st.metric("Quality", "✅ Good", help=f"Silhouette: {silhouette:.3f}")
        elif silhouette > 0.3:
            st.metric("Quality", "⚠️ Fair", help=f"Silhouette: {silhouette:.3f}")
        else:
            st.metric("Quality", "❌ Poor", help=f"Silhouette: {silhouette:.3f}")
    
    st.markdown("---")
    
    # PCA Visualization
    st.markdown("### 📊 Cluster Visualization (PCA)")
    
    # Perform PCA
    pca = PCA(n_components=min(3, len(feature_cols)))
    X_pca = pca.fit_transform(X_scaled)
    
    data_filtered['PC1'] = X_pca[:, 0]
    data_filtered['PC2'] = X_pca[:, 1]
    if X_pca.shape[1] > 2:
        data_filtered['PC3'] = X_pca[:, 2]
    
    # Explained variance
    explained_var = pca.explained_variance_ratio_
    
    st.info(f"""
    **PCA Explained Variance:** 
    - PC1: {explained_var[0]*100:.1f}%
    - PC2: {explained_var[1]*100:.1f}%
    {f"- PC3: {explained_var[2]*100:.1f}%" if len(explained_var) > 2 else ""}
    - Total: {sum(explained_var)*100:.1f}%
    """)
    
    # 2D scatter plot
    fig_pca = px.scatter(
        data_filtered,
        x='PC1',
        y='PC2',
        color='cluster',
        hover_data=['facility_name', 'region', 'facility_type', 'total_emissions', 'pue', 'cfe_pct'],
        title='Facility Clusters (PCA Projection)',
        labels={'cluster': 'Cluster'},
        color_continuous_scale='viridis' if n_clusters > 10 else None,
        category_orders={'cluster': sorted(data_filtered['cluster'].unique())}
    )
    
    fig_pca.update_traces(marker=dict(size=12, line=dict(width=1, color='white')))
    
    fig_pca.update_layout(
        height=500,
        template='plotly_white',
        xaxis_title=f'PC1 ({explained_var[0]*100:.1f}% variance)',
        yaxis_title=f'PC2 ({explained_var[1]*100:.1f}% variance)'
    )
    
    st.plotly_chart(fig_pca, use_container_width=True)
    
    st.markdown("---")
    
    # Cluster Profiles
    st.markdown("### 📋 Cluster Profiles & Characteristics")
    
    # Define cluster characteristics based on metrics
    cluster_profiles = {}
    
    for cluster_id in sorted(data_filtered['cluster'].unique()):
        cluster_data = data_filtered[data_filtered['cluster'] == cluster_id]
        
        # Calculate cluster characteristics
        avg_emissions = cluster_data['total_emissions'].mean()
        avg_pue = cluster_data['pue'].mean()
        avg_cfe = cluster_data['cfe_pct'].mean()
        avg_emissions_intensity = cluster_data['emissions_per_mwh'].mean()
        avg_water_intensity = cluster_data['water_per_mwh'].mean()
        
        # Determine cluster type
        if avg_pue < 1.10 and avg_cfe > 0.65:
            cluster_type = "Best-in-Class"
            strategy = "Maintain excellence, share best practices across portfolio"
            priority = "Focus on water efficiency and waste circularity"
            color = "#27ae60"
        elif avg_pue > 1.12 and avg_emissions_intensity > 300:
            cluster_type = "Inefficient Operations"
            strategy = "HVAC optimization, hot aisle containment, server virtualization"
            priority = "PUE improvement from {:.2f} → 1.08".format(avg_pue)
            color = "#e74c3c"
        elif avg_cfe < 0.30 and avg_emissions_intensity > 250:
            cluster_type = "Dirty Grid, Mixed Ops"
            strategy = "Renewable PPAs, on-site solar, battery storage"
            priority = "Increase CFE from {:.0f}% → 70%+".format(avg_cfe*100)
            color = "#f39c12"
        elif avg_water_intensity > 100:
            cluster_type = "Water-Intensive"
            strategy = "Closed-loop cooling, water recycling systems"
            priority = "Reduce water intensity by 30%"
            color = "#3498db"
        else:
            cluster_type = "Moderate Performance"
            strategy = "Balanced approach: efficiency + renewables"
            priority = "Target 10-15% improvement across all metrics"
            color = "#95a5a6"
        
        # Estimate ROI
        if cluster_type == "Inefficient Operations":
            roi_estimate = "PUE improvements: $1.5M/year savings, 2-3 year payback"
            capital_req = "$5-8M for HVAC upgrades"
        elif cluster_type == "Dirty Grid, Mixed Ops":
            roi_estimate = "Renewable PPAs: Break-even at $50/tonne carbon price"
            capital_req = "$0 upfront (PPA structure), or $20-30M for on-site solar"
        elif cluster_type == "Water-Intensive":
            roi_estimate = "$200-400K/year savings + risk mitigation"
            capital_req = "$0.5-2M for closed-loop systems"
        elif cluster_type == "Best-in-Class":
            roi_estimate = "Incremental gains, focus on innovation"
            capital_req = "$100-500K for pilot programs"
        else:
            roi_estimate = "Mixed portfolio, 3-5 year payback on efficiency"
            capital_req = "$2-5M blended investment"
        
        cluster_profiles[cluster_id] = {
            'name': cluster_type,
            'facilities': len(cluster_data),
            'strategy': strategy,
            'priority': priority,
            'roi': roi_estimate,
            'capital': capital_req,
            'color': color,
            'avg_emissions': avg_emissions,
            'avg_pue': avg_pue,
            'avg_cfe': avg_cfe * 100,
            'avg_emissions_intensity': avg_emissions_intensity,
            'avg_water_intensity': avg_water_intensity
        }
    
    # Display cluster cards
    for cluster_id, profile in cluster_profiles.items():
        with st.expander(f"**Cluster {cluster_id}: {profile['name']}** ({profile['facilities']} facilities)", expanded=True):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Characteristics:**")
                st.markdown(f"- Avg Emissions: {profile['avg_emissions']:,.0f} tonnes/month")
                st.markdown(f"- Avg PUE: {profile['avg_pue']:.3f}")
                st.markdown(f"- Avg CFE: {profile['avg_cfe']:.0f}%")
                st.markdown(f"- Emissions Intensity: {profile['avg_emissions_intensity']:.0f} kg/MWh")
                st.markdown(f"- Water Intensity: {profile['avg_water_intensity']:.0f} gal/MWh")
                
                st.markdown(f"**Decarbonization Strategy:**")
                st.markdown(f"- {profile['strategy']}")
                
                st.markdown(f"**Priority Action:**")
                st.markdown(f"- {profile['priority']}")
            
            with col2:
                st.markdown(f"**Financial Analysis:**")
                st.markdown(f"**Capital Required:**")
                st.markdown(f"{profile['capital']}")
                st.markdown(f"**Expected ROI:**")
                st.markdown(f"{profile['roi']}")
            
            # Show facilities in cluster
            cluster_facilities = data_filtered[data_filtered['cluster'] == cluster_id][
                ['facility_name', 'region', 'facility_type', 'total_emissions', 'pue', 'cfe_pct']
            ].sort_values('total_emissions', ascending=False)
            
            st.markdown(f"**Facilities in Cluster {cluster_id}:**")
            st.dataframe(
                cluster_facilities.style.format({
                    'total_emissions': '{:,.0f}',
                    'pue': '{:.3f}',
                    'cfe_pct': '{:.1%}'
                }),
                use_container_width=True
            )
    
    st.markdown("---")
    
    # Cluster comparison
    st.markdown("### 📊 Cluster Comparison")
    
    cluster_summary = []
    for cluster_id, profile in cluster_profiles.items():
        cluster_summary.append({
            'Cluster': f"{cluster_id}: {profile['name']}",
            'Facilities': profile['facilities'],
            'Avg Emissions': profile['avg_emissions'],
            'Avg PUE': profile['avg_pue'],
            'Avg CFE %': profile['avg_cfe'],
            'Priority': profile['priority']
        })
    
    summary_df = pd.DataFrame(cluster_summary)
    
    # Visualize comparison
    fig_comparison = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Avg Emissions', 'Avg PUE', 'Avg CFE %')
    )
    
    colors_list = [cluster_profiles[i]['color'] for i in sorted(cluster_profiles.keys())]
    
    fig_comparison.add_trace(
        go.Bar(
            x=[f"C{i}" for i in sorted(cluster_profiles.keys())],
            y=[cluster_profiles[i]['avg_emissions'] for i in sorted(cluster_profiles.keys())],
            marker_color=colors_list,
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig_comparison.add_trace(
        go.Bar(
            x=[f"C{i}" for i in sorted(cluster_profiles.keys())],
            y=[cluster_profiles[i]['avg_pue'] for i in sorted(cluster_profiles.keys())],
            marker_color=colors_list,
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig_comparison.add_trace(
        go.Bar(
            x=[f"C{i}" for i in sorted(cluster_profiles.keys())],
            y=[cluster_profiles[i]['avg_cfe'] for i in sorted(cluster_profiles.keys())],
            marker_color=colors_list,
            showlegend=False
        ),
        row=1, col=3
    )
    
    fig_comparison.update_yaxes(title_text="Tonnes CO₂e", row=1, col=1)
    fig_comparison.update_yaxes(title_text="PUE", row=1, col=2)
    fig_comparison.update_yaxes(title_text="CFE %", row=1, col=3)
    
    fig_comparison.update_layout(height=400, template='plotly_white', showlegend=False)
    
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    st.markdown("---")
    
    # Benchmarking within clusters
    st.markdown("### 🎯 Facility Benchmarking")
    st.markdown("Compare individual facilities against their cluster peers")
    
    selected_facility = st.selectbox(
        "Select Facility",
        options=data_filtered['facility_name'].tolist(),
        help="Choose a facility to see how it compares to its peer group"
    )
    
    facility_row = data_filtered[data_filtered['facility_name'] == selected_facility].iloc[0]
    facility_cluster = facility_row['cluster']
    cluster_peers = data_filtered[data_filtered['cluster'] == facility_cluster]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"**{selected_facility}**")
        st.markdown(f"- Region: {facility_row['region']}")
        st.markdown(f"- Type: {facility_row['facility_type']}")
        st.markdown(f"- Cluster: {facility_cluster} ({cluster_profiles[facility_cluster]['name']})")
    
    with col2:
        st.markdown("**Performance vs Cluster Avg:**")
        
        pue_diff = ((facility_row['pue'] - cluster_peers['pue'].mean()) / cluster_peers['pue'].mean()) * 100
        cfe_diff = facility_row['cfe_pct'] - cluster_peers['cfe_pct'].mean()
        emissions_diff = ((facility_row['total_emissions'] - cluster_peers['total_emissions'].mean()) / cluster_peers['total_emissions'].mean()) * 100
        
        if pue_diff < -5:
            st.success(f"✅ PUE: {pue_diff:+.1f}% (Better)")
        elif pue_diff > 5:
            st.error(f"❌ PUE: {pue_diff:+.1f}% (Worse)")
        else:
            st.info(f"➖ PUE: {pue_diff:+.1f}% (Similar)")
        
        if cfe_diff > 0.05:
            st.success(f"✅ CFE: {cfe_diff:+.1%} (Better)")
        elif cfe_diff < -0.05:
            st.error(f"❌ CFE: {cfe_diff:+.1%} (Worse)")
        else:
            st.info(f"➖ CFE: {cfe_diff:+.1%} (Similar)")
    
    with col3:
        # Rank within cluster
        cluster_peers_sorted = cluster_peers.sort_values('emissions_per_mwh')
        rank = (cluster_peers_sorted['facility_name'] == selected_facility).argmax() + 1
        total = len(cluster_peers)
        
        st.metric("Cluster Rank (by intensity)", f"{rank} of {total}")
        
        if rank <= total * 0.33:
            st.success("✅ Top performer in cluster")
        elif rank <= total * 0.67:
            st.info("ℹ️ Middle performer in cluster")
        else:
            st.warning("⚠️ Bottom performer in cluster - opportunity for improvement")
    
    # Spider chart comparison
    fig_spider = go.Figure()
    
    # Normalize metrics for spider chart
    metrics_normalized = {
        'PUE': 1 - (facility_row['pue'] - 1.0) / 0.5,  # Lower is better, normalize around 1.0-1.5
        'CFE %': facility_row['cfe_pct'],
        'Waste Div %': facility_row['waste_diversion_pct'],
        'Emissions/MWh': 1 - min(facility_row['emissions_per_mwh'] / 500, 1),  # Lower is better
        'Water/MWh': 1 - min(facility_row['water_per_mwh'] / 200, 1)  # Lower is better
    }
    
    cluster_normalized = {
        'PUE': 1 - (cluster_peers['pue'].mean() - 1.0) / 0.5,
        'CFE %': cluster_peers['cfe_pct'].mean(),
        'Waste Div %': cluster_peers['waste_diversion_pct'].mean(),
        'Emissions/MWh': 1 - min(cluster_peers['emissions_per_mwh'].mean() / 500, 1),
        'Water/MWh': 1 - min(cluster_peers['water_per_mwh'].mean() / 200, 1)
    }
    
    categories = list(metrics_normalized.keys())
    
    fig_spider.add_trace(go.Scatterpolar(
        r=list(metrics_normalized.values()),
        theta=categories,
        fill='toself',
        name=selected_facility,
        line_color='blue'
    ))
    
    fig_spider.add_trace(go.Scatterpolar(
        r=list(cluster_normalized.values()),
        theta=categories,
        fill='toself',
        name='Cluster Average',
        line_color='red',
        opacity=0.6
    ))
    
    fig_spider.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title=f'{selected_facility} vs Cluster {facility_cluster} Average',
        height=500
    )
    
    st.plotly_chart(fig_spider, use_container_width=True)
    
    st.markdown("---")
    
    # Investment prioritization
    st.markdown("### 💰 Investment Prioritization by Cluster")
    
    investment_summary = []
    
    for cluster_id, profile in cluster_profiles.items():
        cluster_data = data_filtered[data_filtered['cluster'] == cluster_id]
        
        # Estimate total potential impact
        total_emissions = cluster_data['total_emissions'].sum() * 12  # Annual
        
        # Estimate reduction potential based on cluster type
        if profile['name'] == "Inefficient Operations":
            reduction_potential = 0.15  # 15% from efficiency
            cost_per_tonne = 30
        elif profile['name'] == "Dirty Grid, Mixed Ops":
            reduction_potential = 0.25  # 25% from renewables
            cost_per_tonne = 45
        elif profile['name'] == "Water-Intensive":
            reduction_potential = 0.05  # 5% indirect from water efficiency
            cost_per_tonne = 60
        elif profile['name'] == "Best-in-Class":
            reduction_potential = 0.03  # 3% incremental
            cost_per_tonne = 100
        else:
            reduction_potential = 0.10
            cost_per_tonne = 50
        
        annual_reduction = total_emissions * reduction_potential
        
        # Parse capital requirement (rough estimate)
        capital_str = profile['capital']
        if "M" in capital_str:
            # Extract midpoint of range
            import re
            numbers = re.findall(r'\d+', capital_str)
            if len(numbers) >= 2:
                capital_estimate = (float(numbers[0]) + float(numbers[1])) / 2
            elif len(numbers) == 1:
                capital_estimate = float(numbers[0])
            else:
                capital_estimate = 1
        else:
            capital_estimate = 0.5
        
        investment_summary.append({
            'Cluster': f"{cluster_id}: {profile['name']}",
            'Facilities': profile['facilities'],
            'Annual Emissions': total_emissions,
            'Reduction Potential': f"{reduction_potential*100:.0f}%",
            'Annual Reduction': annual_reduction,
            'Capital Required ($M)': capital_estimate,
            'Cost per Tonne': cost_per_tonne,
            'Priority Score': annual_reduction / capital_estimate if capital_estimate > 0 else 0
        })
    
    invest_df = pd.DataFrame(investment_summary)
    invest_df = invest_df.sort_values('Priority Score', ascending=False)
    
    st.dataframe(
        invest_df.style.format({
            'Annual Emissions': '{:,.0f}',
            'Annual Reduction': '{:,.0f}',
            'Capital Required ($M)': '{:.1f}',
            'Cost per Tonne': '${:.0f}',
            'Priority Score': '{:.0f}'
        }).background_gradient(subset=['Priority Score'], cmap='RdYlGn'),
        use_container_width=True
    )
    
    st.info("""
    **Priority Score** = Annual tonnes CO₂e reduced / Capital required ($M)
    
    Higher score = More cost-effective investment (tonnes per million dollars invested)
    """)
    
    st.markdown("---")
    
    # Key takeaways
    st.markdown("### 🎯 Strategic Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Quick Wins (High ROI):**")
        
        high_roi_clusters = invest_df.nlargest(2, 'Priority Score')
        
        for _, row in high_roi_clusters.iterrows():
            st.success(f"""
            **{row['Cluster']}**
            - {row['Annual Reduction']:,.0f} tonnes/year reduction
            - ${row['Capital Required ($M)']:.1f}M investment
            - Priority Score: {row['Priority Score']:.0f}
            """)
    
    with col2:
        st.markdown("**Long-term Strategic Investments:**")
        
        strategic_clusters = invest_df[invest_df['Cost per Tonne'] > 60]
        
        if len(strategic_clusters) > 0:
            for _, row in strategic_clusters.iterrows():
                st.info(f"""
                **{row['Cluster']}**
                - Higher cost (${row['Cost per Tonne']:.0f}/tonne)
                - Strategic value beyond ROI
                - Consider co-benefits (water, risk, ESG)
                """)
        else:
            st.success("✅ All clusters show strong financial ROI!")

except Exception as e:
    st.error(f"Error: {e}")
    import traceback
    st.code(traceback.format_exc())