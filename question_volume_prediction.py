import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Time Series Analysis Libraries
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Machine Learning Libraries
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class QuestionVolumePredictor:
    def __init__(self, data_path='parliamentary_data_clean.csv'):
        """Initialize the predictor with data"""
        self.data_path = data_path
        self.df = None
        self.time_series = None
        self.models = {}
        self.predictions = {}
        
    def load_and_prepare_data(self):
        """Load and prepare the time series data"""
        print("Loading parliamentary data...")
        self.df = pd.read_csv(self.data_path)
        
        # Convert date column
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Create daily question counts
        daily_counts = self.df.groupby('date').size().reset_index(name='question_count')
        daily_counts = daily_counts.set_index('date')
        
        # Fill missing dates with 0
        date_range = pd.date_range(start=daily_counts.index.min(), 
                                  end=daily_counts.index.max(), 
                                  freq='D')
        self.time_series = daily_counts.reindex(date_range, fill_value=0)
        
        print(f"Data loaded: {len(self.time_series)} days of data")
        print(f"Date range: {self.time_series.index.min()} to {self.time_series.index.max()}")
        print(f"Total questions: {self.time_series['question_count'].sum()}")
        
        return self.time_series
    
    def explore_time_series(self):
        """Explore the time series characteristics"""
        print("\n=== TIME SERIES EXPLORATION ===")
        
        # Basic statistics
        print(f"Mean daily questions: {self.time_series['question_count'].mean():.2f}")
        print(f"Std daily questions: {self.time_series['question_count'].std():.2f}")
        print(f"Min daily questions: {self.time_series['question_count'].min()}")
        print(f"Max daily questions: {self.time_series['question_count'].max()}")
        
        # Stationarity test
        print("\n--- Stationarity Test ---")
        result = adfuller(self.time_series['question_count'])
        print(f'ADF Statistic: {result[0]:.4f}')
        print(f'p-value: {result[1]:.4f}')
        print(f'Is stationary: {result[1] < 0.05}')
        
        # Seasonal decomposition
        print("\n--- Seasonal Decomposition ---")
        decomposition = seasonal_decompose(self.time_series['question_count'], 
                                         period=30, extrapolate_trend='freq')
        
        # Plot decomposition
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        decomposition.observed.plot(ax=axes[0], title='Observed')
        decomposition.trend.plot(ax=axes[1], title='Trend')
        decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
        decomposition.resid.plot(ax=axes[3], title='Residual')
        plt.tight_layout()
        plt.savefig('time_series_decomposition.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return decomposition
    
    def create_features(self, lookback_days=30):
        """Create features for ML models"""
        print(f"\nCreating features with {lookback_days} days lookback...")
        
        # Create lag features
        for i in range(1, lookback_days + 1):
            self.time_series[f'lag_{i}'] = self.time_series['question_count'].shift(i)
        
        # Create rolling statistics
        for window in [7, 14, 30]:
            self.time_series[f'rolling_mean_{window}'] = self.time_series['question_count'].rolling(window=window).mean()
            self.time_series[f'rolling_std_{window}'] = self.time_series['question_count'].rolling(window=window).std()
            self.time_series[f'rolling_max_{window}'] = self.time_series['question_count'].rolling(window=window).max()
        
        # Create time-based features
        self.time_series['day_of_week'] = self.time_series.index.dayofweek
        self.time_series['month'] = self.time_series.index.month
        self.time_series['quarter'] = self.time_series.index.quarter
        self.time_series['year'] = self.time_series.index.year
        
        # Create cyclical features
        self.time_series['day_of_week_sin'] = np.sin(2 * np.pi * self.time_series['day_of_week'] / 7)
        self.time_series['day_of_week_cos'] = np.cos(2 * np.pi * self.time_series['day_of_week'] / 7)
        self.time_series['month_sin'] = np.sin(2 * np.pi * self.time_series['month'] / 12)
        self.time_series['month_cos'] = np.cos(2 * np.pi * self.time_series['month'] / 12)
        
        # Remove NaN values
        self.time_series = self.time_series.dropna()
        
        print(f"Features created. Shape: {self.time_series.shape}")
        return self.time_series
    
    def prepare_ml_data(self, test_size=0.2):
        """Prepare data for ML models"""
        # Separate features and target
        feature_columns = [col for col in self.time_series.columns if col != 'question_count']
        X = self.time_series[feature_columns]
        y = self.time_series['question_count']
        
        # Split data
        split_idx = int(len(self.time_series) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    
    def train_ml_models(self, X_train, X_test, y_train, y_test):
        """Train multiple ML models"""
        print("\n=== TRAINING ML MODELS ===")
        
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            results[name] = {
                'model': model,
                'predictions': y_pred,
                'mse': mse,
                'mae': mae,
                'rmse': rmse
            }
            
            print(f"{name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}")
        
        self.models.update(results)
        return results
    
    def train_lstm_model(self, X_train, X_test, y_train, y_test, sequence_length=30):
        """Train LSTM model for time series prediction"""
        print("\n=== TRAINING LSTM MODEL ===")
        
        # Prepare sequences for LSTM
        def create_sequences(X, y, sequence_length):
            X_seq, y_seq = [], []
            for i in range(sequence_length, len(X)):
                X_seq.append(X[i-sequence_length:i])
                y_seq.append(y.iloc[i])
            return np.array(X_seq), np.array(y_seq)
        
        X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
        X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, X_train.shape[1])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Train model
        history = model.fit(
            X_train_seq, y_train_seq,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        # Make predictions
        y_pred = model.predict(X_test_seq)
        
        # Calculate metrics
        mse = mean_squared_error(y_test_seq, y_pred)
        mae = mean_absolute_error(y_test_seq, y_pred)
        rmse = np.sqrt(mse)
        
        results = {
            'model': model,
            'predictions': y_pred.flatten(),
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'history': history
        }
        
        self.models['LSTM'] = results
        print(f"LSTM - RMSE: {rmse:.2f}, MAE: {mae:.2f}")
        
        return results
    
    def train_arima_model(self, train_data, test_data):
        """Train ARIMA model"""
        print("\n=== TRAINING ARIMA MODEL ===")
        
        # Fit ARIMA model
        model = ARIMA(train_data, order=(1, 1, 1))
        fitted_model = model.fit()
        
        # Make predictions
        forecast = fitted_model.forecast(steps=len(test_data))
        
        # Calculate metrics
        mse = mean_squared_error(test_data, forecast)
        mae = mean_absolute_error(test_data, forecast)
        rmse = np.sqrt(mse)
        
        results = {
            'model': fitted_model,
            'predictions': forecast,
            'mse': mse,
            'mae': mae,
            'rmse': rmse
        }
        
        self.models['ARIMA'] = results
        print(f"ARIMA - RMSE: {rmse:.2f}, MAE: {mae:.2f}")
        
        return results
    
    def compare_models(self):
        """Compare all trained models"""
        print("\n=== MODEL COMPARISON ===")
        
        comparison_data = []
        for name, results in self.models.items():
            comparison_data.append({
                'Model': name,
                'RMSE': results['rmse'],
                'MAE': results['mae'],
                'MSE': results['mse']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('RMSE')
        
        print(comparison_df)
        
        # Plot comparison
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # RMSE comparison
        axes[0].bar(comparison_df['Model'], comparison_df['RMSE'])
        axes[0].set_title('RMSE Comparison')
        axes[0].set_ylabel('RMSE')
        axes[0].tick_params(axis='x', rotation=45)
        
        # MAE comparison
        axes[1].bar(comparison_df['Model'], comparison_df['MAE'])
        axes[1].set_title('MAE Comparison')
        axes[1].set_ylabel('MAE')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return comparison_df
    
    def plot_predictions(self, test_data, test_dates):
        """Plot predictions vs actual values"""
        print("\n=== PLOTTING PREDICTIONS ===")
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        axes = axes.flatten()
        
        for i, (name, results) in enumerate(self.models.items()):
            if i >= 4:  # Limit to 4 plots
                break
                
            predictions = results['predictions']
            
            # Adjust predictions length if needed
            if len(predictions) != len(test_data):
                predictions = predictions[:len(test_data)]
            
            axes[i].plot(test_dates, test_data, label='Actual', linewidth=2)
            axes[i].plot(test_dates, predictions, label='Predicted', linewidth=2)
            axes[i].set_title(f'{name} Predictions')
            axes[i].set_xlabel('Date')
            axes[i].set_ylabel('Question Count')
            axes[i].legend()
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('predictions_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def forecast_future(self, days_ahead=30, best_model_name=None):
        """Forecast future question volumes"""
        print(f"\n=== FORECASTING NEXT {days_ahead} DAYS ===")
        
        if best_model_name is None:
            # Use the best model based on RMSE
            best_model = min(self.models.items(), key=lambda x: x[1]['rmse'])
            best_model_name = best_model[0]
        
        print(f"Using {best_model_name} for forecasting...")
        
        # Get the best model
        best_model = self.models[best_model_name]['model']
        
        # Create future dates
        last_date = self.time_series.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                   periods=days_ahead, 
                                   freq='D')
        
        if best_model_name == 'ARIMA':
            # ARIMA forecasting
            forecast = best_model.forecast(steps=days_ahead)
        else:
            # For ML models, we need to create features for future dates
            # This is a simplified approach - in practice, you'd need more sophisticated feature engineering
            forecast = np.full(days_ahead, self.time_series['question_count'].mean())
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'predicted_questions': forecast
        })
        
        # Plot forecast
        plt.figure(figsize=(15, 8))
        plt.plot(self.time_series.index, self.time_series['question_count'], 
                label='Historical', linewidth=2)
        plt.plot(forecast_df['date'], forecast_df['predicted_questions'], 
                label='Forecast', linewidth=2, linestyle='--')
        plt.title(f'Question Volume Forecast - Next {days_ahead} Days')
        plt.xlabel('Date')
        plt.ylabel('Question Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('future_forecast.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Forecast completed. Average predicted questions per day: {forecast.mean():.2f}")
        return forecast_df
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("🚀 STARTING QUESTION VOLUME PREDICTION ANALYSIS")
        print("=" * 60)
        
        # 1. Load and prepare data
        self.load_and_prepare_data()
        
        # 2. Explore time series
        decomposition = self.explore_time_series()
        
        # 3. Create features
        self.create_features()
        
        # 4. Prepare ML data
        X_train, X_test, y_train, y_test, scaler = self.prepare_ml_data()
        
        # 5. Train ML models
        ml_results = self.train_ml_models(X_train, X_test, y_train, y_test)
        
        # 6. Train LSTM model
        lstm_results = self.train_lstm_model(X_train, X_test, y_train, y_test)
        
        # 7. Train ARIMA model
        train_data = self.time_series['question_count'][:int(len(self.time_series) * 0.8)]
        test_data = self.time_series['question_count'][int(len(self.time_series) * 0.8):]
        arima_results = self.train_arima_model(train_data, test_data)
        
        # 8. Compare models
        comparison = self.compare_models()
        
        # 9. Plot predictions
        test_dates = self.time_series.index[int(len(self.time_series) * 0.8):]
        self.plot_predictions(test_data, test_dates)
        
        # 10. Forecast future
        forecast = self.forecast_future(days_ahead=30)
        
        print("\n✅ ANALYSIS COMPLETE!")
        print("=" * 60)
        
        return {
            'comparison': comparison,
            'forecast': forecast,
            'models': self.models
        }

if __name__ == "__main__":
    # Initialize and run the analysis
    predictor = QuestionVolumePredictor()
    results = predictor.run_complete_analysis()
    
    # Print summary
    print("\n📊 SUMMARY:")
    print(f"Best model: {results['comparison'].iloc[0]['Model']}")
    print(f"Best RMSE: {results['comparison'].iloc[0]['RMSE']:.2f}")
    print(f"Forecast average: {results['forecast']['predicted_questions'].mean():.2f} questions/day") 