import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys
sys.path.append('..')
from utils.data_loader import load_emissions_data, load_combined_metrics, load_targets

st.set_page_config(page_title="AI Forecasting", page_icon="🤖", layout="wide")

st.markdown('<style>.main-header {font-size: 2.5rem; font-weight: 700; color: #9b59b6;}</style>', unsafe_allow_html=True)
st.markdown('<p class="main-header">🤖 AI-Powered Forecasting</p>', unsafe_allow_html=True)

st.markdown("""
## Predictive Analytics for Emissions & Operational Metrics

Use advanced time series models to forecast future emissions and operational performance 
under different scenarios.

**Models Available:**
- **Prophet**: Facebook's robust forecasting with seasonality and trends
- **ARIMA/SARIMA**: Classical statistical forecasting with customizable parameters
- **Holt-Winters**: Exponential smoothing with seasonal components
- **Scenario Analysis**: Business-as-usual vs decarbonization strategies

**ARIMA Tuning:**
- Use **Recommended Defaults** for quick, reliable results
- Use **Custom Parameters** to experiment and optimize via AIC/BIC scores
""")

st.markdown("---")

# Load data
@st.cache_data
def load_data():
    """Load emissions and operational data"""
    emissions = load_emissions_data()
    combined = load_combined_metrics()
    return emissions, combined

try:
    emissions_data, combined_data = load_data()
    
    if len(emissions_data) == 0:
        st.error("No data available")
        st.stop()
    
    st.success(f"✅ Loaded {len(emissions_data):,} records for forecasting")
    
    # Sidebar Configuration
    st.sidebar.header("⚙️ Forecast Configuration")
    
    # Target selection
    forecast_target_options = {
        'Total Emissions': ('emissions', 'total_emissions'),
        'Scope 1 Emissions': ('emissions', 'scope1_tonnes'),
        'Scope 2 Emissions': ('emissions', 'scope2_market_tonnes'),
        'Scope 3 Emissions': ('emissions', 'scope3_tonnes'),
        'PUE (Efficiency)': ('operational', 'pue'),
        'Water Consumption': ('operational', 'water_consumption_gallons'),
        'Carbon-Free Energy %': ('operational', 'cfe_pct'),
        'Waste Diversion %': ('operational', 'waste_diversion_pct')
    }
    
    forecast_target = st.sidebar.selectbox(
        "Forecast Target",
        options=list(forecast_target_options.keys()),
        help="Choose which metric to forecast"
    )
    
    data_source, target_col = forecast_target_options[forecast_target]
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["Prophet", "ARIMA", "Holt-Winters"],
        help="Select forecasting algorithm"
    )

    # ARIMA parameters (only show if ARIMA selected)
    if model_type == "ARIMA":
        st.sidebar.markdown("---")
        st.sidebar.markdown("**ARIMA Configuration**")
        
        # Offer preset or custom
        arima_mode = st.sidebar.radio(
            "Parameter Selection",
            ["Recommended Defaults", "Custom Parameters"],
            help="Use proven defaults or tune manually"
        )
        
        if arima_mode == "Custom Parameters":
            st.sidebar.markdown("**Non-Seasonal (p, d, q):**")
            arima_p = st.sidebar.slider("p (AR order)", 0, 5, 1, 
                                       help="Autoregressive order - how many past values")
            arima_d = st.sidebar.slider("d (Differencing)", 0, 2, 1, 
                                       help="Differencing order - make data stationary")
            arima_q = st.sidebar.slider("q (MA order)", 0, 5, 1, 
                                       help="Moving average order - past forecast errors")
            
            st.sidebar.markdown("**Seasonal (P, D, Q, s):**")
            seasonal_p = st.sidebar.slider("P (Seasonal AR)", 0, 2, 1)
            seasonal_d = st.sidebar.slider("D (Seasonal Diff)", 0, 1, 1)
            seasonal_q = st.sidebar.slider("Q (Seasonal MA)", 0, 2, 1)
            seasonal_m = 12  # Monthly data
        else:
            # Recommended defaults for business time series
            arima_p, arima_d, arima_q = 1, 1, 1
            seasonal_p, seasonal_d, seasonal_q = 1, 1, 1
            seasonal_m = 12
            
            st.sidebar.success("Using: ARIMA(1,1,1)×(1,1,1,12)")
            st.sidebar.info("Good for most business data with monthly seasonality")
    
    # Forecast horizon
    forecast_months = st.sidebar.slider(
        "Forecast Horizon (months)",
        min_value=6,
        max_value=60,
        value=24,
        step=6,
        help="How far into the future to predict"
    )
    
    # Scenario selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Scenario Analysis")
    
    scenario = st.sidebar.selectbox(
        "Decarbonization Scenario",
        [
            "Business as Usual",
            "Aggressive Decarbonization",
            "Efficiency Focus",
            "Combined Strategy"
        ],
        help="Apply different growth assumptions to the forecast"
    )
    
    # Prepare data based on target
    if data_source == 'emissions':
        df = emissions_data.copy()
    else:
        df = combined_data.copy()
    
    # Aggregate to monthly
    monthly_data = df.groupby(pd.Grouper(key='date', freq='M')).agg({
        target_col: 'mean' if target_col in ['pue', 'cfe_pct', 'waste_diversion_pct'] else 'sum'
    }).reset_index()
    
    monthly_data = monthly_data.dropna()
    
    if len(monthly_data) < 12:
        st.error("Need at least 12 months of data for forecasting")
        st.stop()
    
    # Split train/test
    test_size = st.sidebar.slider("Test Set Size (months)", 3, 12, 6, 1)
    train_data = monthly_data[:-test_size]
    test_data = monthly_data[-test_size:]
    
    st.markdown("---")
    
    # Display current metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_value = monthly_data[target_col].iloc[-1]
        if target_col in ['pue', 'cfe_pct', 'waste_diversion_pct']:
            st.metric("Current Value", f"{current_value:.2f}")
        else:
            st.metric("Current Value", f"{current_value:,.0f}")
    
    with col2:
        avg_value = monthly_data[target_col].mean()
        if target_col in ['pue', 'cfe_pct', 'waste_diversion_pct']:
            st.metric("Historical Avg", f"{avg_value:.2f}")
        else:
            st.metric("Historical Avg", f"{avg_value:,.0f}")
    
    with col3:
        # YoY change
        if len(monthly_data) >= 13:
            yoy_change = ((current_value - monthly_data[target_col].iloc[-13]) / monthly_data[target_col].iloc[-13]) * 100
            st.metric("YoY Change", f"{yoy_change:+.1f}%")
        else:
            st.metric("YoY Change", "N/A")
    
    with col4:
        # Trend
        z = np.polyfit(range(len(monthly_data)), monthly_data[target_col], 1)
        trend = "Increasing" if z[0] > 0 else "Decreasing"
        st.metric("Trend", trend)
    
    st.markdown("---")
    
    # Train models
    st.markdown("### 📈 Model Training & Forecasting")
    
    with st.spinner("Training model..."):
        
        if model_type == "Prophet":
            # Prepare data for Prophet
            prophet_train = pd.DataFrame({
                'ds': train_data['date'],
                'y': train_data[target_col]
            })
            
            prophet_test = pd.DataFrame({
                'ds': test_data['date'],
                'y': test_data[target_col]
            })
            
            # Train Prophet
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            model.fit(prophet_train)
            
            # Predict on test set
            test_predictions = model.predict(prophet_test[['ds']])
            test_pred_values = test_predictions['yhat'].values
            
            # Forecast future
            future_dates = model.make_future_dataframe(periods=forecast_months, freq='M')
            forecast = model.predict(future_dates)
            
        elif model_type == "ARIMA":
            try:
                # Display model configuration
                st.info(f"📊 Training SARIMA({arima_p},{arima_d},{arima_q})×({seasonal_p},{seasonal_d},{seasonal_q},{seasonal_m})")
                
                # Fit SARIMAX model
                model = SARIMAX(
                    train_data[target_col],
                    order=(arima_p, arima_d, arima_q),
                    seasonal_order=(seasonal_p, seasonal_d, seasonal_q, seasonal_m),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                
                model_fit = model.fit(disp=False, maxiter=200)
                
                # Display model diagnostics
                with st.expander("📈 Model Diagnostics", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("AIC", f"{model_fit.aic:.0f}", 
                                 help="Akaike Information Criterion - Lower is better")
                    
                    with col2:
                        st.metric("BIC", f"{model_fit.bic:.0f}",
                                 help="Bayesian Information Criterion - Lower is better")
                    
                    with col3:
                        st.metric("Log Likelihood", f"{model_fit.llf:.0f}")
                    
                    st.markdown("**Interpretation:**")
                    st.markdown("- Lower AIC/BIC = Better model fit")
                    st.markdown("- Compare different parameter combinations")
                    st.markdown("- Typical AIC range: 500-5000 depending on data scale")
                
                # Predict on test set
                test_pred_values = model_fit.forecast(steps=len(test_data))
                
                # Forecast future
                forecast_steps = len(test_data) + forecast_months
                forecast_result = model_fit.get_forecast(steps=forecast_steps)
                forecast_values = forecast_result.predicted_mean
                forecast_ci = forecast_result.conf_int()
                
                # Create forecast dataframe
                future_dates = pd.date_range(
                    start=train_data['date'].iloc[-1],
                    periods=forecast_steps + 1,
                    freq='M'
                )[1:]
                
                forecast = pd.DataFrame({
                    'ds': future_dates,
                    'yhat': forecast_values.values,
                    'yhat_lower': forecast_ci.iloc[:, 0].values,
                    'yhat_upper': forecast_ci.iloc[:, 1].values
                })
                
            except Exception as e:
                st.error(f"❌ ARIMA model failed: {str(e)}")
                st.warning("**Troubleshooting tips:**")
                st.markdown("- Try different (p,d,q) values")
                st.markdown("- Reduce differencing (d) if data is already stationary")
                st.markdown("- Start with simpler model: (1,1,1)×(0,0,0,0)")
                st.markdown("- Or switch to Prophet/Holt-Winters")
                st.stop()
        
        else:  # Holt-Winters
            # Train Holt-Winters
            try:
                model = ExponentialSmoothing(
                    train_data[target_col],
                    seasonal_periods=12,
                    trend='add',
                    seasonal='add'
                ).fit()
                
                # Predict on test set
                test_pred_values = model.forecast(steps=len(test_data))
                
                # Forecast future
                forecast_values = model.forecast(steps=len(test_data) + forecast_months)
                
                # Create forecast dataframe
                future_dates = pd.date_range(
                    start=train_data['date'].iloc[-1],
                    periods=len(forecast_values) + 1,
                    freq='M'
                )[1:]
                
                forecast = pd.DataFrame({
                    'ds': future_dates,
                    'yhat': forecast_values,
                    'yhat_lower': forecast_values * 0.9,
                    'yhat_upper': forecast_values * 1.1
                })
                
            except Exception as e:
                st.error(f"Holt-Winters failed: {e}. Try Prophet or ARIMA instead.")
                st.stop()
    
    # Apply scenario adjustments
    scenario_adjustments = {
        'Business as Usual': 1.0,
        'Aggressive Decarbonization': 0.85,  # 15% reduction
        'Efficiency Focus': 0.90,  # 10% reduction
        'Combined Strategy': 0.80  # 20% reduction
    }
    
    # Only apply to emissions forecasts
    if data_source == 'emissions' and scenario != 'Business as Usual':
        adjustment = scenario_adjustments[scenario]
        
        # Apply gradual adjustment over forecast period
        future_forecast = forecast[forecast['ds'] > test_data['date'].max()].copy()
        adjustment_schedule = np.linspace(1.0, adjustment, len(future_forecast))
        
        forecast.loc[forecast['ds'] > test_data['date'].max(), 'yhat'] *= adjustment_schedule
        forecast.loc[forecast['ds'] > test_data['date'].max(), 'yhat_lower'] *= adjustment_schedule
        forecast.loc[forecast['ds'] > test_data['date'].max(), 'yhat_upper'] *= adjustment_schedule
    
    # For operational metrics, apply improvements
    if data_source == 'operational':
        if target_col == 'pue' and scenario != 'Business as Usual':
            # Improve PUE over time
            future_forecast = forecast[forecast['ds'] > test_data['date'].max()].copy()
            improvement_per_month = 0.001 if scenario == 'Efficiency Focus' else 0.0005
            improvements = np.arange(len(future_forecast)) * improvement_per_month
            forecast.loc[forecast['ds'] > test_data['date'].max(), 'yhat'] -= improvements
            
        elif target_col == 'cfe_pct' and scenario == 'Aggressive Decarbonization':
            # Accelerate CFE growth toward 100%
            future_forecast = forecast[forecast['ds'] > test_data['date'].max()].copy()
            current_cfe = monthly_data[target_col].iloc[-1]
            months_to_100 = min(60, forecast_months)  # 5 years to reach 100%
            monthly_increase = (1.0 - current_cfe) / months_to_100
            increases = np.minimum(1.0, current_cfe + np.arange(len(future_forecast)) * monthly_increase)
            forecast.loc[forecast['ds'] > test_data['date'].max(), 'yhat'] = increases
    
    st.success("✅ Model trained successfully!")
    
    # Calculate test set metrics
    test_mae = mean_absolute_error(test_data[target_col], test_pred_values)
    test_rmse = np.sqrt(mean_squared_error(test_data[target_col], test_pred_values))
    test_mape = np.mean(np.abs((test_data[target_col] - test_pred_values) / test_data[target_col])) * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Test MAE", f"{test_mae:,.0f}")
    
    with col2:
        st.metric("Test RMSE", f"{test_rmse:,.0f}")
    
    with col3:
        st.metric("Test MAPE", f"{test_mape:.1f}%")
    
    st.markdown("---")
    
    # Visualization
    st.markdown(f"### 📊 {forecast_target} Forecast")
    st.markdown(f"**Scenario:** {scenario}")
    
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=monthly_data['date'],
        y=monthly_data[target_col],
        mode='lines+markers',
        name='Historical',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))
    
    # Test predictions
    fig.add_trace(go.Scatter(
        x=test_data['date'],
        y=test_pred_values,
        mode='markers',
        name='Test Predictions',
        marker=dict(color='orange', size=8, symbol='x')
    ))
    
    # Future forecast
    future_forecast = forecast[forecast['ds'] > test_data['date'].max()]
    
    fig.add_trace(go.Scatter(
        x=future_forecast['ds'],
        y=future_forecast['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='red', width=3, dash='dash')
    ))
    
    # Confidence intervals
    fig.add_trace(go.Scatter(
        x=future_forecast['ds'],
        y=future_forecast['yhat_upper'],
        mode='lines',
        name='Upper Bound',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=future_forecast['ds'],
        y=future_forecast['yhat_lower'],
        mode='lines',
        name='Confidence Interval',
        line=dict(width=0),
        fillcolor='rgba(255, 0, 0, 0.2)',
        fill='tonexty'
    ))
    
    # Add target line if applicable
    if data_source == 'emissions':
        try:
            targets = load_targets()
            if len(targets) > 0:
                # Find relevant target
                if 'Scope 1' in forecast_target or 'Scope 2' in forecast_target:
                    target_row = targets[targets['scope'] == 'Scope 1 & 2'].iloc[0]
                elif 'Scope 3' in forecast_target:
                    target_row = targets[targets['scope'] == 'Scope 3'].iloc[0]
                else:
                    target_row = targets[targets['scope'] == 'Scope 1 & 2'].iloc[0]
                
                target_year = target_row['target_year']
                target_emissions = target_row['target_emissions']
                
                # Adjust target for specific scope if needed
                if 'Scope 1' in forecast_target:
                    target_emissions *= 0.05  # Scope 1 is ~5% of total
                elif 'Scope 2' in forecast_target:
                    target_emissions *= 0.20  # Scope 2 is ~20% of total
                elif 'Scope 3' in forecast_target:
                    target_emissions *= 0.75  # Scope 3 is ~75% of total
                
                fig.add_hline(
                    y=target_emissions / 12,  # Monthly target
                    line_dash='dot',
                    line_color='green',
                    annotation_text=f'{target_year} Target',
                    annotation_position='right'
                )
        except:
            pass
    
    # Special targets for operational metrics
    if target_col == 'pue':
        fig.add_hline(y=1.08, line_dash='dot', line_color='green', 
                     annotation_text='Industry Best Practice (1.08)')
    elif target_col == 'cfe_pct':
        fig.add_hline(y=1.0, line_dash='dot', line_color='green',
                     annotation_text='100% CFE Goal')
    elif target_col == 'waste_diversion_pct':
        fig.add_hline(y=0.90, line_dash='dot', line_color='green',
                     annotation_text='90% Diversion Goal')
    
    fig.update_layout(
        title=f'{forecast_target} - Historical & Forecast ({scenario})',
        xaxis_title='Date',
        yaxis_title=forecast_target,
        height=500,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    st.plotly_chart(fig, width = 'stretch')
    
    # Forecast statistics
    st.markdown("---")
    st.markdown("### 📊 Forecast Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Key forecast values
        forecast_end = future_forecast['yhat'].iloc[-1]
        current_end = monthly_data[target_col].iloc[-1]
        change = forecast_end - current_end
        pct_change = (change / current_end) * 100 if current_end != 0 else 0
        
        st.markdown(f"**Forecast Period:** {forecast_months} months")
        if target_col in ['pue', 'cfe_pct', 'waste_diversion_pct']:
            st.markdown(f"**Current Value:** {current_end:.3f}")
            st.markdown(f"**Forecast End Value:** {forecast_end:.3f}")
            st.markdown(f"**Expected Change:** {change:+.3f} ({pct_change:+.1f}%)")
        else:
            st.markdown(f"**Current Value:** {current_end:,.0f}")
            st.markdown(f"**Forecast End Value:** {forecast_end:,.0f}")
            st.markdown(f"**Expected Change:** {change:+,.0f} ({pct_change:+.1f}%)")
    
    with col2:
        # Scenario impact
        if scenario != 'Business as Usual':
            st.markdown(f"**Scenario Applied:** {scenario}")
            st.markdown(f"**Adjustment Factor:** {scenario_adjustments.get(scenario, 1.0):.0%}")
            
            if data_source == 'emissions':
                reduction_pct = (1 - scenario_adjustments[scenario]) * 100
                st.success(f"✅ {reduction_pct:.0f}% reduction vs business-as-usual")
            elif target_col == 'pue' and scenario == 'Efficiency Focus':
                st.success(f"✅ Accelerated PUE improvement (0.001/month)")
            elif target_col == 'cfe_pct' and scenario == 'Aggressive Decarbonization':
                st.success(f"✅ Accelerated CFE growth toward 100%")
    
    # Scenario comparison
    if data_source == 'emissions':
        st.markdown("---")
        st.markdown("### 🔄 Scenario Comparison")
        
        # Run all scenarios
        scenario_results = {}
        
        for scen_name, adjustment in scenario_adjustments.items():
            # Quick forecast for comparison
            if model_type == "Prophet":
                scen_forecast = model.predict(future_dates)
            else:
                scen_forecast = forecast.copy()
            
            # Apply adjustment
            future_mask = scen_forecast['ds'] > test_data['date'].max()
            if scen_name != 'Business as Usual':
                adjustment_schedule = np.linspace(1.0, adjustment, future_mask.sum())
                scen_forecast.loc[future_mask, 'yhat'] *= adjustment_schedule
            
            scenario_results[scen_name] = scen_forecast[future_mask]['yhat'].values
        
        # Plot comparison
        fig_comparison = go.Figure()
        
        colors = {
            'Business as Usual': '#95a5a6',
            'Aggressive Decarbonization': '#27ae60',
            'Efficiency Focus': '#3498db',
            'Combined Strategy': '#9b59b6'
        }
        
        for scen_name, values in scenario_results.items():
            fig_comparison.add_trace(go.Scatter(
                x=future_forecast['ds'],
                y=values,
                mode='lines',
                name=scen_name,
                line=dict(color=colors[scen_name], width=3)
            ))
        
        fig_comparison.update_layout(
            title='Emissions Forecast Under Different Scenarios',
            xaxis_title='Date',
            yaxis_title=forecast_target,
            height=400,
            template='plotly_white',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_comparison, width = 'stretch')
        
        # Cumulative reduction comparison
        st.markdown("#### 📉 Cumulative Emissions Reduction")
        
        bau_total = scenario_results['Business as Usual'].sum()
        
        reduction_summary = []
        for scen_name, values in scenario_results.items():
            scen_total = values.sum()
            reduction = bau_total - scen_total
            reduction_pct = (reduction / bau_total) * 100
            
            reduction_summary.append({
                'Scenario': scen_name,
                'Total Emissions': f"{scen_total:,.0f}",
                'Reduction vs BAU': f"{reduction:,.0f} ({reduction_pct:.1f}%)"
            })
        
        reduction_df = pd.DataFrame(reduction_summary)
        st.dataframe(reduction_df, width="stretch")
    
    st.markdown("---")
    
    # Insights and recommendations
    st.markdown("### 💡 Key Insights & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Model Performance:**")
        if test_mape < 10:
            st.success(f"✅ Excellent accuracy (MAPE: {test_mape:.1f}%)")
        elif test_mape < 20:
            st.info(f"ℹ️ Good accuracy (MAPE: {test_mape:.1f}%)")
        else:
            st.warning(f"⚠️ Moderate accuracy (MAPE: {test_mape:.1f}%) - Consider more data or features")
        
        st.markdown("**Forecast Confidence:**")
        uncertainty = (future_forecast['yhat_upper'] - future_forecast['yhat_lower']).mean()
        uncertainty_pct = (uncertainty / future_forecast['yhat'].mean()) * 100
        
        if uncertainty_pct < 20:
            st.success(f"✅ High confidence (±{uncertainty_pct:.0f}%)")
        elif uncertainty_pct < 40:
            st.info(f"ℹ️ Moderate confidence (±{uncertainty_pct:.0f}%)")
        else:
            st.warning(f"⚠️ Lower confidence (±{uncertainty_pct:.0f}%) - High variability")
    
    with col2:
        st.markdown("**Strategic Implications:**")
        
        if data_source == 'emissions':
            if pct_change > 10:
                st.warning(f"⚠️ Emissions projected to increase {pct_change:.0f}% - intervention needed")
            elif pct_change > 0:
                st.info(f"ℹ️ Emissions growing {pct_change:.0f}% - consider acceleration of decarbonization")
            else:
                st.success(f"✅ Emissions declining {abs(pct_change):.0f}% - on track")
        
        elif target_col == 'pue':
            if forecast_end < 1.10:
                st.success(f"✅ PUE improving to {forecast_end:.3f} - industry-leading")
            else:
                st.info(f"ℹ️ PUE at {forecast_end:.3f} - opportunity for efficiency improvements")
        
        elif target_col == 'cfe_pct':
            if forecast_end > 0.80:
                st.success(f"✅ CFE reaching {forecast_end*100:.0f}% - excellent progress")
            else:
                st.info(f"ℹ️ CFE at {forecast_end*100:.0f}% - consider additional renewable contracts")

except Exception as e:
    st.error(f"Error: {e}")
    import traceback
    st.code(traceback.format_exc())