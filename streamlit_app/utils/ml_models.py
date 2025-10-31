import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy import stats


"""
ML utility functions for analytics
"""

class EmissionsForecaster:
    """Time series forecasting of emissions data, Streamlit Page 3"""

    def __init__(self, method='prophet'):
        self.method = method
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, df, date_col='date', target_col='emissions'):
        """
        Fit forecasting model
        
        Args:
            df: DataFrame with date and emissions columns
            date_col: Name of date column
            target_col: Name of target column
        """

        if self.method == 'prophet':
            #prepare data for Prophet
            prophet_df = df[[date_col, target_col]].copy()
            prophet_df.columns = ['ds', 'y']

            #initialize and fit prophet model
            self.model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='multiplicative',
                changepoint_prior_scale=0.05,
                interval_width=0.95
            )

            #add quarterly seasonality for business relevance
            self.model.add_seasonality(
                name='quarterly',
                period=91.25,
                fourier_order=5
            )

            self.model.fit(prophet_df)

        elif self.method == 'holt_winters':
            #holt-winters exponential smoothing
            self.model = ExponentialSmoothing(
                df[target_col],
                seasonal_periods=12,
                trend='add',
                seasonal='add',
                damped_trend=True
            ).fit()

        return self
    
    def predict(self, periods=12, freq='M'):
        """
        Generate forecast

        Args:
            periods: number of periods to forecast
            freq: frequency, M for monthly, D for daily

        Returns:
            DataFrame with forecast and confidence intervals
        """

        if self.method == 'prophet':
            future = self.model.make_future_dataframe(periods=periods, freq=freq)
            forecast = self.model.predict(future)
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend','yearly']]
        
        elif self.method == 'holt_winters':
            forecast = self.model.forecast(steps=periods)
            return pd.DataFrame({
                'forecast': forecast.values,
                'lower': forecast.values * 0.9,
                'upper': forecast.values * 1.1
            })
        
        elif self.method == 'arima':
            forecast = self.model.forecast(steps=periods)

            last_date = self.data['ds'].max()
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(months = 1),
                periods = periods,
                freq = freq
            )

            #confidence intervals
            std_error = np.std(self.data['y'].values[-12:])

            return {
                'dates': future_dates,
                'forecast': forecast.values,
                'upper': forecast.values + 1.96 * std_error, # 95% CI
                'lower': forecast.values - 1.96 * std_error,
                'components': None
            }
        
    def get_components(self):
        """Get trend and seasonal components"""
        if self.method == 'prophet' and self.model:
            return self.model
        return None