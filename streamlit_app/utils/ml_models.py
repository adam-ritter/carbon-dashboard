import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy import stats
import shap


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
                seasonal_period=12,
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
    

class AnomalyDetector:
    """Detect anomalies in emissions data"""

    def __init__(self, method='isolation_forest', contamination=0.05):
        self.method = method
        self.contamination = contamination
        self.model = None

    def fit_predict(self, df, features):
        """
        Detect anomalies
        
        Args:
            df: DataFrame with emissions data
            features: list of feature column names
            
        Returns:
            DataFrame with anomaly flags and sources
        """

        X = df[features].fillna(0)

        if self.method == 'isolation_forest':
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )

            df['anomaly_flag'] = self.model.fit_predict(X)
            df['anomaly_score'] = self.model.score_samples(X)

        elif self.method == 'zscore':
            #statistical method using z-scores
            z_scores = np.abs(stats.zscore(X, axis=0))
            df['anomaly_flag'] = (z_scores > 3).any(axis=1).astype(int) * -1 + 1
            df['anomaly_score'] = -zscores.max(axis=1)

        return df
    
    def get_anomalies(self, df):
        """Return df with only identified anomaly data points"""
        return df[df['anomaly_flag'] == -1].copy()
    

class EmissionsDriverAnalyzer:
    """Identify business drivers of emissions"""

    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.features_names = None

    def fit(self, X, y, models = None):
        """
        Train multiple regression models

        Args:
            X: Feature DataFrame
            y: Target series (emissions)
            models: dict of models to train (optional)
        """

        self.feature_names = X.columns.tolist()

        #default models if None provided
        if models is None:
            models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(alpha=1.0),
                'Random Forest': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42
                )
            }

        #split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = 0.2, random_state=42
        )

        #scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        #train each model
        results = {}

        for name, model in models.items():
            model.fit(X_train_scaled, y_train)

            #cross-validation score
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train,
                cv=5, scoring='r2'
            )

            #test score
            test_score = model.score(X_test_scaled, y_test)

            #predictions for metrics
            y_pred = model.predict(X_test_scaled)
            mae = np.mean(np.abs(y_test - y_pred))
            rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

            results[name] = {
                'model': model,
                'cv_score_mean': cv_scores.mean(),
                'cv_score_std': cv_scores.std(),
                'test_r2': test_score,
                'mae': mae,
                'rmse': rmse,
                'X_test': X_test,
                'X_test_scaled': X_test_scaled,
                'y_test': y_test,
                'y_pred': y_pred
            }

        self.models = results

        #identify best model

        self.best_model_name = max(results, key=lambda x: results[x]['test_r2'])
        self.best_model = results[self.best_model_name]['model']

        return self
    
    def get_feature_importance(self):
        """get feature important from best model"""
        if self.best_model_name in ['Random Forest', 'Gradient Boosting']:
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending = False)

            return importance_df
        else:
            #for linear models, use abs coefficients
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': np.abs(self.best_model.coef_)
            }).sort_values('importance', ascending = False)

            return importance_df
        
    def get_shap_values(self):
        """calculate SHAP values for model explainability"""
        if self.best_model_name not in self.models:
            return None
        
        X_test_scaled = self.models[self.best_model_name]['X_test_scaled']
        X_test = self.models[self.best_model_name]['X_test']

        try:
            explainer = shap.Explainer(self.best_model, X_test_scaled)
            shap_values = explainer(X_test_scaled)

            return shap_values, X_test
        except Exception as e:
            print(f"SHAP calculation failed: {e}")
            return None, None
        
    def get_model_comparison(self):
        """Return comparison of all models"""
        comparison = pd.DataFrame({
            'Model': list(self.models.keys()),
            'CV R2 (mean)': [m['cv_score_mean'] for m in self.models.values()],
            'CV R2 (std)': [m['cv_score_std'] for m in self.models.values()],
            'Test R2': [m['test_r2'] for m in self.models.values()],
            'MAE': [m['mae'] for m in self.models.values()],
            'RMSE': [m['rmse'] for m in self.models.values()]
        })

        return comparison
    

class FacilityClusterer:
    """Segment facilities, unsupervised"""

    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters
        self.model = None
        self.scaler = StandardScaler()
        self.pca = None
        self.feature_names = None

    def fit_predict(self, df, features):
        """
        Cluster facilities
        
        Args:
            df: DataFrame with facility data
            features: list of feature column names
            
        Returns:
            DataFrame with cluster assignments
        """

        self.feature_names = features
        X = df[features].fillna(0)

        #scale features
        X_scaled = self.scaler.fit_transform(X)

        #K-means cluster
        self.model = KMeans(
            n_clusters = self.n_clusters,
            random_state = 42,
            n_init = 10
        )

        df['cluster'] = self.model.fit_predict(X_scaled)

        #PCA for visualization
        self.pca = PCA(n_components = 2)
        pca_features = self.pca.fit_transform(X_scaled)

        df['pca1'] = pca_features[:,0]
        df['pca2'] = pca_features[:,1]

        return df
    
    def get_cluster_profiles(self, df, features):
        """get statistical profile of each cluster"""

        profiles = df.groupby('cluster')[features].agg(['mean', 'std', 'count'])
        return profiles
    
    def get_elbow_data(self, df, features, k_range=range(2,10)):
        """calculate inertia for elbow method"""
        X = df[features].fillna(0)
        X_scaled = self.scaler.fit_transform(X)

        inertias = []
        for k in k_range:
            kmeans = KMeans(n_clusters = k, random_state = 42, n_init = 10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)

        return list(k_range), inertias
    
    def get_pca_variance(self):
        """get explained variance from PCA"""
        if self.pca:
            return self.pca.explained_variance_ratio_
        return None
    
    def calculate_emission_intensity(emissions, revenue):
        """calculate emission intensity metrics"""
        return emissions / revenue if revenue > 0 else 0
    
    def detect_trend(series, method = 'linear'):
        """detect trend direction and magnitude"""
        x = np.arange(len(series))
        y = series.values

        if method == 'linear':
            slope, intercept = np.polyfit(x, y, 1)
            return slope, intercept
        
        return None
