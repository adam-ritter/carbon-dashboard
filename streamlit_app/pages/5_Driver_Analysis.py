import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
sys.path.append('..')
from utils.ml_models import EmissionsDriverAnalyzer
import sqlite3
import shap
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

st.set_page_config(page_title="Driver Analysis", page_icon="📈", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #28a745;
        margin-bottom: 1rem;
    }
    .driver-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .insight-box {
        background-color: #f0fff4;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .recommendation {
        background-color: #e6f7ff;
        border-left: 5px solid #1890ff;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">📈 ML-Based Emissions Driver Analysis</p>', unsafe_allow_html=True)

st.markdown("""
## Identify Business Drivers of Emissions

Using machine learning regression models to quantify which business metrics drive emissions:
- **Multiple ML Models**: Compare Linear, Ridge, Random Forest, Gradient Boosting
- **Feature Importance**: Rank drivers by impact
- **SHAP Values**: Explain individual predictions with model-agnostic approach
- **Actionable Insights**: Data-driven decarbonization strategies

**Business Value:** Focus reduction efforts on highest-impact drivers, set targets aligned with business metrics.
""")

# Sidebar controls
st.sidebar.header("🎛️ Analysis Settings")

target_scope = st.sidebar.selectbox(
    "Emissions Scope to Analyze",
    ['Scope 1 & 2 (Operational)', 'Scope 1', 'Scope 2', 'Scope 3', 'Total'],
    help="Which emissions to analyze"
)

include_features = st.sidebar.multiselect(
    "Business Metrics to Include",
    ['Revenue', 'Headcount', 'Square Feet', 'Server Count', 'Production Volume', 'Renewable Energy %'],
    default=['Revenue', 'Headcount', 'Square Feet', 'Server Count', 'Renewable Energy %'],
    help="Select business drivers to analyze"
)

model_selection = st.sidebar.multiselect(
    "ML Models to Train",
    ['Linear Regression', 'Ridge Regression', 'Random Forest', 'Gradient Boosting'],
    default=['Linear Regression', 'Random Forest', 'Gradient Boosting'],
    help="Select regression models to compare"
)

# Load data
@st.cache_data
def load_driver_data():
    """Load emissions data with business metrics"""
    conn = sqlite3.connect('../data/sustainability_data.db')
    
    query = """
    SELECT 
        e.date,
        e.facility_id,
        f.facility_name,
        f.region,
        SUM(e.scope1_tonnes) as scope1,
        SUM(e.scope2_market_tonnes) as scope2,
        SUM(e.scope3_tonnes) as scope3,
        SUM(e.scope1_tonnes + e.scope2_market_tonnes) as scope12,
        SUM(e.scope1_tonnes + e.scope2_market_tonnes + e.scope3_tonnes) as total_emissions,
        AVG(b.revenue_millions) as revenue_millions,
        AVG(b.headcount) as headcount,
        AVG(b.square_feet) as square_feet,
        AVG(b.server_count) as server_count,
        AVG(b.production_volume) as production_volume,
        AVG(e.renewable_pct) as renewable_energy_pct
    FROM emissions_monthly e
    JOIN facilities f ON e.facility_id = f.facility_id
    LEFT JOIN business_metrics b ON e.facility_id = b.facility_id AND e.date = b.date
    GROUP BY e.date, e.facility_id
    HAVING AVG(b.revenue_millions) IS NOT NULL
    ORDER BY e.date
    """
    
    df = pd.read_sql_query(query, conn)
    df['date'] = pd.to_datetime(df['date'])
    conn.close()
    
    return df

try:
    data = load_driver_data()
    
    # Data summary
    st.subheader("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Data Points",
            f"{len(data):,}",
            help="Monthly facility observations"
        )
    
    with col2:
        st.metric(
            "Facilities",
            data['facility_id'].nunique(),
            help="Unique facilities in analysis"
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
            help="Average monthly emissions"
        )
    
    st.markdown("---")
    
    # Prepare data for modeling
    st.subheader("🔧 Model Configuration")
    
    # Map target selection
    target_map = {
        'Scope 1 & 2 (Operational)': 'scope12',
        'Scope 1': 'scope1',
        'Scope 2': 'scope2',
        'Scope 3': 'scope3',
        'Total': 'total_emissions'
    }
    
    target_col = target_map[target_scope]
    
    # Map feature selection
    feature_map = {
        'Revenue': 'revenue_millions',
        'Headcount': 'headcount',
        'Square Feet': 'square_feet',
        'Server Count': 'server_count',
        'Production Volume': 'production_volume',
        'Renewable Energy %': 'renewable_energy_pct'
    }
    
    feature_cols = [feature_map[f] for f in include_features]
    
    # Prepare X and y
    X = data[feature_cols].fillna(0)
    y = data[target_col]
    
    # Display data sample
    with st.expander("🔍 View Data Sample"):
        sample_data = data[['date', 'facility_name'] + feature_cols + [target_col]].head(10)
        st.dataframe(
            sample_data.style.format({
                col: '{:,.2f}' if col in feature_cols else '{:,.0f}'
                for col in sample_data.columns if col not in ['date', 'facility_name']
            }),
            width = 'stretch'
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Target Variable:** {target_scope}  
        **Observations:** {len(X):,}  
        **Features:** {len(feature_cols)}
        """)
    
    with col2:
        # Correlation preview
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        st.metric(
            "Strongest Correlation",
            f"{correlations.index[0]}: {correlations.iloc[0]:.3f}",
            help="Preliminary correlation analysis"
        )
    
    st.markdown("---")
    
    # Train models
    st.subheader("🤖 Model Training & Comparison")
    
    # Map model names
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    
    model_map = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    }
    
    selected_models = {name: model_map[name] for name in model_selection}
    
    if len(selected_models) == 0:
        st.warning("⚠️ Please select at least one model in the sidebar")
        st.stop()
    
    with st.spinner("Training machine learning models..."):
        analyzer = EmissionsDriverAnalyzer()
        analyzer.fit(X, y, models=selected_models)
    
    st.success(f"✅ Trained {len(selected_models)} models successfully!")
    
    # Model comparison table
    comparison = analyzer.get_model_comparison()
    
    st.subheader("📊 Model Performance Comparison")
    
    # Highlight best model
    best_model_idx = comparison['Test R2'].idxmax()
    
    def highlight_best(row):
        if row.name == best_model_idx:
            return ['background-color: #c6efce'] * len(row)
        return [''] * len(row)
    
    st.dataframe(
        comparison.style
            .format({
                'CV R² (mean)': '{:.3f}',
                'CV R² (std)': '{:.3f}',
                'Test R2': '{:.3f}',
                'MAE': '{:,.0f}',
                'RMSE': '{:,.0f}'
            })
            .apply(highlight_best, axis=1)
            .background_gradient(subset=['Test R2'], cmap='Greens'),
        width = 'stretch'
    )
    
    best_model_name = comparison.loc[best_model_idx, 'Model']
    best_r2 = comparison.loc[best_model_idx, 'Test R2']
    
    st.info(f"""
    **🏆 Best Model:** {best_model_name}  
    **R² Score:** {best_r2:.3f} ({best_r2*100:.1f}% of variance explained)  
    **Interpretation:** The model can predict {best_r2*100:.0f}% of emission variations based on business metrics.
    """)
    
    # Actual vs Predicted plot
    col1, col2 = st.columns(2)
    
    with col1:
        # Scatter plot
        y_test = analyzer.models[best_model_name]['y_test']
        y_pred = analyzer.models[best_model_name]['y_pred']
        
        fig_scatter = go.Figure()
        
        fig_scatter.add_trace(go.Scatter(
            x=y_test,
            y=y_pred,
            mode='markers',
            marker=dict(
                size=8,
                color=y_test,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Actual")
            ),
            name='Predictions',
            text=[f"Actual: {a:,.0f}<br>Predicted: {p:,.0f}" for a, p in zip(y_test, y_pred)],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        
        fig_scatter.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash', width=2),
            name='Perfect Prediction',
            showlegend=True
        ))
        
        fig_scatter.update_layout(
            title='Actual vs Predicted Emissions',
            xaxis_title='Actual Emissions (tonnes CO₂e)',
            yaxis_title='Predicted Emissions (tonnes CO₂e)',
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_scatter, width = 'stretch')
    
    with col2:
        # Residuals plot
        residuals = y_test - y_pred
        
        fig_residuals = go.Figure()
        
        fig_residuals.add_trace(go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            marker=dict(
                size=8,
                color=np.abs(residuals),
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Abs Error")
            ),
            text=[f"Predicted: {p:,.0f}<br>Error: {r:+,.0f}" for p, r in zip(y_pred, residuals)],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Zero line
        fig_residuals.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
            annotation_text="Zero Error"
        )
        
        fig_residuals.update_layout(
            title='Residuals Plot',
            xaxis_title='Predicted Emissions (tonnes CO₂e)',
            yaxis_title='Residual (Actual - Predicted)',
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_residuals, width = 'stretch')
    
    st.markdown("---")
    
    # Feature importance
    st.subheader("🎯 Feature Importance Analysis")
    
    feature_importance = analyzer.get_feature_importance()
    
    # Reverse map feature names
    reverse_feature_map = {v: k for k, v in feature_map.items()}
    feature_importance['feature_display'] = feature_importance['feature'].map(reverse_feature_map)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Bar chart
        fig_importance = go.Figure()
        
        fig_importance.add_trace(go.Bar(
            x=feature_importance['importance'],
            y=feature_importance['feature_display'],
            orientation='h',
            marker=dict(
                color=feature_importance['importance'],
                colorscale='Viridis',
                showscale=False
            ),
            text=feature_importance['importance'].round(3),
            textposition='outside'
        ))
        
        fig_importance.update_layout(
            title=f'Feature Importance - {best_model_name}',
            xaxis_title='Importance Score',
            yaxis_title='Business Metric',
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_importance, width = 'stretch')
    
    with col2:
        st.markdown("### 📊 Rankings")
        
        for idx, row in feature_importance.head(5).iterrows():
            pct = row['importance'] / feature_importance['importance'].sum() * 100
            st.metric(
                f"{idx+1}. {row['feature_display']}",
                f"{pct:.1f}%",
                help=f"Contributes {pct:.1f}% to model predictions"
            )
    
    # Key driver insights
    top_driver = feature_importance.iloc[0]
    
    st.markdown(f"""
    <div class="insight-box">
    <h4>💡 Primary Emission Driver</h4>
    <p><strong>{top_driver['feature_display']}</strong> is the strongest predictor of {target_scope.lower()}, 
    explaining {top_driver['importance']/feature_importance['importance'].sum()*100:.0f}% of the model's predictive power.</p>
    
    <p><strong>What this means:</strong> Changes in {top_driver['feature_display'].lower()} have the largest impact 
    on emissions. Focus decarbonization efforts here for maximum ROI.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # SHAP Analysis
    st.subheader("🔬 SHAP Analysis - Model Explainability")
    
    st.markdown("""
    **SHAP (SHapley Additive exPlanations)** values show how each feature contributes to individual predictions:
    - **Red dots**: High feature values
    - **Blue dots**: Low feature values
    - **Position on x-axis**: Impact on prediction (positive = increases emissions, negative = decreases)
    """)
    
    with st.spinner("Calculating SHAP values... (this may take a minute)"):
        try:
            shap_values, X_test = analyzer.get_shap_values()
            
            if shap_values is not None:
                # SHAP summary plot
                fig_shap, ax = plt.subplots(figsize=(10, 6))
                
                # Rename columns for display
                X_test_display = X_test.copy()
                X_test_display.columns = [reverse_feature_map.get(col, col) for col in X_test.columns]
                
                shap.summary_plot(
                    shap_values.values,
                    X_test_display,
                    show=False,
                    plot_size=(10, 6)
                )
                
                st.pyplot(fig_shap, width = 'stretch')
                
                # SHAP waterfall for a single prediction
                st.markdown("#### 🌊 SHAP Waterfall - Sample Prediction")
                
                st.markdown("""
                Select a data point to see how each feature contributed to its prediction:
                """)
                
                sample_idx = st.slider(
                    "Select Sample Index",
                    min_value=0,
                    max_value=len(X_test)-1,
                    value=0,
                    help="Choose which prediction to explain"
                )
                
                # Waterfall plot
                fig_waterfall, ax_waterfall = plt.subplots(figsize=(10, 6))
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_values.values[sample_idx],
                        base_values=shap_values.base_values[sample_idx],
                        data=X_test.iloc[sample_idx].values,
                        feature_names=[reverse_feature_map.get(col, col) for col in X_test.columns]
                    ),
                    show=False
                )
                st.pyplot(fig_waterfall, width = 'stretch')
                
                # Interpretation
                actual_value = analyzer.models[best_model_name]['y_test'].iloc[sample_idx]
                predicted_value = analyzer.models[best_model_name]['y_pred'][sample_idx]
                
                st.info(f"""
                **Sample Interpretation:**
                - **Actual Emissions:** {actual_value:,.0f} tonnes CO₂e
                - **Predicted Emissions:** {predicted_value:,.0f} tonnes CO₂e
                - **Error:** {actual_value - predicted_value:+,.0f} tonnes CO₂e
                
                The waterfall shows how each feature pushed the prediction up or down from the baseline.
                """)
                
            else:
                st.warning("⚠️ SHAP analysis not available for this model type")
                
        except Exception as e:
            st.error(f"Error calculating SHAP values: {e}")
            st.info("SHAP analysis works best with tree-based models (Random Forest, Gradient Boosting)")
    
    st.markdown("---")
    
    # Scenario analysis
    st.subheader("🎲 Scenario Modeling")
    
    st.markdown("""
    Use the sliders below to model how changes in business metrics affect emissions:
    """)
    
    # Create scenario inputs
    scenario_values = {}
    
    cols = st.columns(3)
    
    for idx, feature in enumerate(feature_cols):
        col_idx = idx % 3
        display_name = reverse_feature_map.get(feature, feature)
        
        with cols[col_idx]:
            avg_value = X[feature].mean()
            min_value = X[feature].min()
            max_value = X[feature].max()
            
            #fix error with step=0
            if max_val>min_val:
                step_size = float((max_value - min_value) / 100)
            else:
                step_size = 0.01

            scenario_values[feature] = st.slider(
                display_name,
                min_value=float(min_value),
                max_value=float(max_value),
                value=float(avg_value),
                step=step_size,
                format="%.2f"
            )
    
    # Make prediction
    scenario_df = pd.DataFrame([scenario_values])
    scenario_scaled = analyzer.scaler.transform(scenario_df)
    scenario_prediction = analyzer.best_model.predict(scenario_scaled)[0]
    
    # Compare to average
    avg_emissions = y.mean()
    change = scenario_prediction - avg_emissions
    change_pct = (change / avg_emissions) * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Predicted Emissions",
            f"{scenario_prediction:,.0f} tonnes",
            help="Based on scenario inputs"
        )
    
    with col2:
        st.metric(
            "vs. Average",
            f"{change_pct:+.1f}%",
            delta=f"{change:+,.0f} tonnes",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            "Historical Average",
            f"{avg_emissions:,.0f} tonnes",
            help="Baseline for comparison"
        )
    
    st.markdown("---")
    
    # Recommendations
    st.subheader("💡 Data-Driven Recommendations")
    
    # Top 3 drivers
    top_3_drivers = feature_importance.head(3)
    
    st.markdown("### 🎯 Priority Actions Based on Driver Analysis")
    
    for idx, row in top_3_drivers.iterrows():
        driver_name = row['feature_display']
        importance = row['importance'] / feature_importance['importance'].sum() * 100
        
        # Generate recommendations based on driver
        if 'Revenue' in driver_name:
            rec = f"""
            **Action:** Decouple emissions from revenue growth
            - Set emissions intensity targets (tonnes CO₂e per $M revenue)
            - Invest in energy efficiency as you scale
            - Procure renewable energy at growing facilities
            """
        elif 'Headcount' in driver_name:
            rec = f"""
            **Action:** Manage per-employee emissions
            - Promote remote work to reduce facility energy
            - Optimize commute patterns (transit, EV charging)
            - Set per-capita emission reduction targets
            """
        elif 'Square Feet' in driver_name:
            rec = f"""
            **Action:** Improve building efficiency
            - Upgrade HVAC systems in existing facilities
            - Implement smart building controls
            - Prioritize energy-efficient design in new construction
            """
        elif 'Server' in driver_name:
            rec = f"""
            **Action:** Optimize data center operations
            - Increase server utilization rates
            - Transition to more efficient hardware
            - Maximize renewable energy procurement
            """
        elif 'Renewable' in driver_name:
            rec = f"""
            **Action:** Accelerate renewable energy adoption
            - Increase renewable energy percentage target
            - Sign additional PPAs or invest in on-site generation
            - Track market-based Scope 2 emissions
            """
        else:
            rec = f"""
            **Action:** Target {driver_name.lower()} for reduction
            - Set specific KPIs for this metric
            - Monitor correlation with emissions monthly
            - Implement efficiency initiatives
            """
        
        st.markdown(f"""
        <div class="recommendation">
        <h4>#{idx+1}: {driver_name} ({importance:.0f}% of model impact)</h4>
        {rec}
        </div>
        """, unsafe_allow_html=True)
    
    # Export functionality
    st.markdown("---")
    st.subheader("💾 Export Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export feature importance
        importance_export = feature_importance[['feature_display', 'importance']].copy()
        importance_export.columns = ['Feature', 'Importance']
        
        csv = importance_export.to_csv(index=False)
        st.download_button(
            label="📥 Download Feature Importance (CSV)",
            data=csv,
            file_name=f"feature_importance_{target_scope.replace(' ', '_')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Export summary report
        report = f"""
EMISSIONS DRIVER ANALYSIS REPORT
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

TARGET: {target_scope}
BEST MODEL: {best_model_name}
R² SCORE: {best_r2:.3f}

FEATURE IMPORTANCE RANKINGS:
{chr(10).join([f"{i+1}. {row['feature_display']}: {row['importance']/feature_importance['importance'].sum()*100:.1f}%" 
               for i, (_, row) in enumerate(top_3_drivers.iterrows())])}

MODEL PERFORMANCE:
- Test R²: {best_r2:.3f}
- MAE: {comparison.loc[best_model_idx, 'MAE']:,.0f} tonnes CO₂e
- RMSE: {comparison.loc[best_model_idx, 'RMSE']:,.0f} tonnes CO₂e

TOP RECOMMENDATIONS:
1. Focus decarbonization efforts on {top_3_drivers.iloc[0]['feature_display']}
2. Monitor {top_3_drivers.iloc[1]['feature_display']} closely for early warnings
3. Set targets aligned with {top_3_drivers.iloc[2]['feature_display']} trajectory

NEXT STEPS:
- Share findings with relevant business units
- Update reduction strategies based on driver analysis
- Re-run analysis quarterly to track changing patterns
        """
        
        st.download_button(
            label="📄 Download Report (TXT)",
            data=report,
            file_name=f"driver_analysis_report_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )

except Exception as e:
    st.error(f"Error loading data or running analysis: {e}")
    st.info("Please ensure the database exists and contains emissions data with business metrics.")
    import traceback
    st.code(traceback.format_exc())