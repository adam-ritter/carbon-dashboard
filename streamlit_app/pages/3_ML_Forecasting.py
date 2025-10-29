import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
sys.path.append('..')
from utils.ml_models import EmissionsForecaster
from utils.data_loader import load_emissions_data
import sqlite3
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error

st.set_page_config(page_title = "ML Forecasting", layout = "wide")

# custom CSS
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
    .insight-box {
        background-color: #f0f8ff;
        border-left: 5px solid #1f77b4;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🤖 AI-Powered Emissions Forecasting</p>', unsafe_allow_html=True)

st.markdown("""
## Machine Learning for Predictive Analytics

Using advanced time series models to forecast future emissions and identify trends:
- **Prophet**: Facebook's forecasting algorithm (handles seasonality, holidays, trend changes)
- **Holt-Winters**: Exponential smoothing for seasonal patterns
- **ARIMA**: Autoregressive integrated moving average (coming soon)

**Business Value:** Enable proactive strategy adjustments, early warning of target misses, and data-driven planning.
""")

# Sidebar controls
st.sidebar.header("🎛️ Forecast Configuration")

forecast_method = st.sidebar.selectbox(
    "Forecasting Method",
    ['prophet', 'holt_winters'],
    help="Prophet is generally more robust for business data"
)

forecast_horizon = st.sidebar.slider(
    "Forecast Horizon (months)",
    min_value=3,
    max_value=24,
    value=12,
    help="Number of months to forecast ahead"
)

scope_selection = st.sidebar.multiselect(
    "Emissions Scope",
    ['Scope 1', 'Scope 2 (Market)', 'Scope 3', 'Total'],
    default=['Total'],
    help="Select which scopes to forecast"
)

confidence_interval = st.sidebar.slider(
    "Confidence Interval",
    min_value=0.80,
    max_value=0.99,
    value=0.95,
    step=0.01,
    help="Width of prediction intervals"
)

#load data
@st.cache_data
def load_data():
    """load emissions time series data"""
    conn = sqlite3.connect('../data/sustainability_data.db')

    query = """
    SELECT
        date,
        SUM(scope1_tonnes) as scope1,
        SUM(scope2_market_tonnes) as scope2_market,
        SUM(scope3_tonnes) as scope3,
        SUM(scope1_tonnes + scope2_market_tonnes + scope3_tonnes) as total_emissions
    FROM emissions_monthly
    GROUP BY date
    ORDER BY date
    """

    df = pd.read_sql_query(query, conn)
    df['date'] = pd.to_datetime(df['date'])
    conn.close()

    return df

try:
    data = load_data()

    #display data summary
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Historical Data",
            len(data),
            help = "Months of historical emissions data"
        )

    with col2:
        latest_emissions = data['total_emissions'].iloc[-1]
        st.metric(
            "Latest Month",
            f"{latest_emissions:,.0f} tonnes",
            help = "Most recent monthly emissions"
        )

    with col3:
        yoy_change = ((data['total_emissions'].iloc[-1] / data['total_emissions'].iloc[-13]) - 1) * 100
        st.metric(
            "YoY Change",
            f"{yoy_change:+.1f}%",
            delta = f"{yoy_change:.1f}%",
            delta_color = "inverse"
        )

    with col4:
        avg_monthly = data['total_emissions'].mean()
        st.metric(
            "Avg Monthly",
            f"{avg_monthly:,.0f} tonnes",
            help = "Average monthly emissions"
        )

    st.markdown("---")

    ###################################################################
    #model training
    ###################################################################
    st.subheader("Forecast Generation")

    #prepare data based on scope selection
    if 'Total' in scope_selection:
        forecast_data = data[['date', 'total_emissions']].copy()
        forecast_data.columns = ['date', 'emissions']
        target_name = 'Total Emissions'
    elif 'Scope 1' in scope_selection:
        forecast_data = data[['date', 'scope1']].copy()
        forecast_data.columns = ['date', 'emissions']
        target_name = 'Scope 1 Emissions'
    elif 'Scope 2 (Market)' in scope_selection:
        forecast_data = data[['date', 'scope2_market']].copy()
        forecast_data.columns = ['date', 'emissions']
        target_name = 'Scope 2 Emissions'
    else:
        forecast_data = data[['date', 'scope3']].copy()
        forecast_data.columns = ['date', 'emissions']
        target_name = 'Scope 3 Emissions'

    #train model
    with st.spinner(f"Training {forecast_method.upper()} model..."):
        forecaster = EmissionsForecaster(method=forecast_method)
        forecaster.fit(forecast_data, date_col='date', target_col='emissions')
        forecast = forecaster.predict(periods = forecast_horizon, freq = 'M')

    st.success(f"Model trained successfully! Generated {forecast_horizon}-month forecast.")

    #visualize
    st.subheader("Forecast Visualization")

    fig = go.Figure()

    #historical data
    fig.add_trace(go.Scatter(
        x=forecast_data['date'],
        y=forecast_data['emissions'],
        mode='lines+markers',
        name='Historical',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6)
    ))

    if forecast_method == 'prophet':
        #forecast line
        future_dates = forecast['ds'][len(forecast_data):]
        future_forecast = forecast['yhat'][len(forecast_data):]

        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_forecast,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            marker=dict(size=6, symbol='diamond')
        ))

        #confidence interval
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_upper'],
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))

        fig.add_trace(go.Scatter(
            x_forecast['ds'],
            y=forecast['yhat_lower'],
            mode='lines',
            name='Lower Bound',
            line=dict(width=0),
            fillcolor='rgba(255, 127, 14, 0.2)',
            fill='tonexty',
            showlegend=True,
            hoverinfo='skip'
        ))

    fig.update_layout(
        title=f'{target_name} - {forecast_horizon}-Month Forecast',
        xaxis_title='Date',
        yaxis_title='Emissions (tonnes CO₂e)',
        hovermode='x unified',
        height=500,
        template='plotly_white',
        legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)
    
    st.plotly_chart(fig, user_container_width = True)

    #model performance metrics
    st.subheader("Model Performance")

    col1, col2, col3 = st.columns(3)

    if forecast_method == 'prophet':
        #calculate metrics on historical data
        historical_forecast = forecast[forecast['ds'].isin(forecast_data['date'])]
        merged = forecast_data.merge(
            historical_forecast[['ds', 'yhat']],
            left_on = 'date',
            right_on = 'ds',
            how = 'inner'
        )

        mape = mean_absolute_percentage_error(merged['emissions'], merged['yhat'])
        mae = mean_absolute_error(merged['emissions'], merged['yhat'])
        rmse = np.sqrt(mean_squared_error(merged['emissions'], merged['yhat']))

        with col1:
            st.metric(
                "MAPE",
                f"{mape:.1%}",
                help="Mean Absolute Percentage Error"
            )
        
        with col2:
            st.metric(
                "MAE",
                f"{mae:,.0f} tonnes",
                help="Mean Absolute Error"
            )
        
        with col3:
            st.metric(
                "RMSE",
                f"{rmse:,.0f} tonnes",
                help="Root Mean Squared Error"
            )

    #forecast table
    st.subheader("Detailed Forecast")

    if forecast_method == 'prophet':
        forecast_display = forecast[forecast['ds'] > forecast_data['date'].max()].copy()
        forecast_display = forecast_display[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(forecast_horizon)
        forecast_display.columns = ['Month', 'Forecast', 'Lower Bound (95%)', 'Upper Bound (95%)']
        forecast_display['Month'] = forecast_display['Month'].dt.strftime('%Y-%m')

        st.dataframe(
            forecast_display.style.format({
                'Forecast': '{:,.0f}',
                'Lower Bound (95%)': '{:,.0f}',
                'Upper Bound (95%)': '{:,.0f}'
            }).background_gradient(subset=['Forecast'], cmap='YlOrRd'),
            use_container_width=True
        )  

    #trends and seasonality
    st.subheader("Trend and Seasonality Analysis")

    if forecast_method == 'prophet':
        col1, col2 = st.columns(2)

        with col1:
            #trend
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['trend'],
                mode='lines',
                name='Trend',
                line=dict(color='#2ca02c', width=3)
            ))

            fig_trend.update_layout(
                title='Long-term Trend',
                xaxis_title='Date',
                yaxis_title='Trend Component',
                height=350,
                template='plotly_white'
            )

            st.plotly_chart(fig_trend, use_container_width = True)

        with col2:
            #yearly seasonality
            if 'yearly' in forecast.columns:
                fig_seasonal = go.Figure()
                fig_seasonal.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yearly'],
                    mode='lines',
                    name='Yearly Seasonality',
                    line=dict(color='#d62728', width=2)
                ))

                fig_seasonal.update_layout(
                    title='Seasonal Pattern',
                    xaxis_title='Date',
                    yaxis_title='Seasonal Component',
                    height=350,
                    template='plotly_white'
                )

                st.plotly_chart(fig_seasonal, use_container_width=True)

    #key insights
    st.subheader("Key Insights and Recommendations")

    #calculate trend direction
    recent_trend = forecast['yhat'].iloc[-1] - forecast['yhat'].iloc[-forecast_horizon]
    trend_direction = "increasing" if recent_trend >0 else "decreasing"
    trend_pct = abs(recent_trend / forecast['yhat'].iloc[-forecast_horizon]) * 100

    #highest/lowest months
    future_forecast_df = forecast[forecast['ds'] > forecast_data['date'].max()].head(forecast_horizon)
    highest_month = future_forecast_df.loc[future_forecast_df['yhat'].idxmax(), 'ds'].strftime('%B %Y')
    lowest_month = future_forecast_df.loc[future_forecast_df['yhat'].idxmin(), 'ds'].strftime('%B %Y')

    st.markdown(f"""
    <div class="insight-box">
    <h4>📈 Trend Analysis</h4>
    <ul>
        <li><strong>Overall Trend:</strong> Emissions are <strong>{trend_direction}</strong> by {trend_pct:.1f}% over the forecast period</li>
        <li><strong>Peak Month:</strong> {highest_month} (highest forecasted emissions)</li>
        <li><strong>Lowest Month:</strong> {lowest_month} (lowest forecasted emissions)</li>
        <li><strong>Forecast Uncertainty:</strong> {'High' if mape > 0.15 else 'Moderate' if mape > 0.08 else 'Low'} (MAPE: {mape:.1%})</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    #recommendations
    st.markdown("""
    ### 🎯 Recommended Actions
    
    **Short-term (1-3 months):**
    - Monitor high-emission months identified in forecast
    - Prepare contingency plans for peak periods
    - Update stakeholders on expected trajectory
    
    **Medium-term (3-6 months):**
    - Adjust reduction initiatives based on trend
    - Reallocate resources to address forecasted hotspots
    - Refine targets if forecast suggests targets at risk
    
    **Long-term (6-12 months):**
    - Develop structural changes if negative trend persists
    - Integrate forecasts into budget planning
    - Set Science-Based Targets aligned with projections
    """)

    # Export functionality
    st.subheader("💾 Export Forecast")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV export
        csv = forecast_display.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast (CSV)",
            data=csv,
            file_name=f"emissions_forecast_{forecast_horizon}m.csv",
            mime="text/csv"
        )
    
    with col2:
        # Summary report
        report = f"""
EMISSIONS FORECAST REPORT
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

Target: {target_name}
Method: {forecast_method.upper()}
Horizon: {forecast_horizon} months
Confidence: {confidence_interval:.0%}

PERFORMANCE METRICS:
- MAPE: {mape:.1%}
- MAE: {mae:,.0f} tonnes CO₂e
- RMSE: {rmse:,.0f} tonnes CO₂e

TREND ANALYSIS:
- Direction: {trend_direction.capitalize()}
- Change: {trend_pct:.1f}%
- Peak: {highest_month}
- Low: {lowest_month}
        """
        
        st.download_button(
            label="📄 Download Report (TXT)",
            data=report,
            file_name=f"forecast_report_{forecast_horizon}m.txt",
            mime="text/plain"
        )

except Exception as e:
    st.error(f"Error loading data or generating forecast: {e}")
    st.info("Please ensure the database exists and contains emissions data.")