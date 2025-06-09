"""
BEAST Trading System - ML Predictor
Machine learning based price predictions and pattern recognition
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

from config.settings import config
from utils.logger import get_logger

logger = get_logger(__name__)

class MLPredictor:
    """
    Machine learning predictions for trading
    Uses ensemble methods for robust predictions
    """
    
    def __init__(self, config):
        self.config = config
        
        # Model settings
        self.model_params = {
            'features': [
                'returns_1h', 'returns_4h', 'returns_24h',
                'volume_ratio', 'rsi', 'macd_signal',
                'bb_position', 'atr_ratio', 'trend_strength',
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
            ],
            'prediction_horizons': [1, 4, 24],  # hours
            'min_train_samples': 1000,
            'update_frequency': 24  # hours
        }
        
        # Models for different timeframes
        self.models = {}
        self.scalers = {}
        self.last_update = {}
        self.model_performance = {}
        
        # Feature importance tracking
        self.feature_importance = {}
        
        # Model directory
        self.model_dir = 'models'
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Load existing models if available
        self._load_models()
        
        logger.info("MLPredictor initialized")
    
    async def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate ML-based predictions
        """
        result = {
            'status': 'predicted',
            'predictions': {},
            'confidence': 0.0,
            'direction': None,
            'prediction_slope': 0.0,
            'timeframe_alignment': 0.0,
            'feature_contributions': {},
            'model_confidence': {}
        }
        
        try:
            # Validate input data
            if 'price_data' not in data or data['price_data'] is None:
                result['status'] = 'no_data'
                return result
            
            df = data['price_data']
            
            # Prepare features
            features_df = self._prepare_features(df)
            if features_df is None or len(features_df) == 0:
                result['status'] = 'insufficient_features'
                return result
            
            # Generate predictions for each horizon
            predictions = {}
            confidences = {}
            
            for horizon in self.model_params['prediction_horizons']:
                pred, conf = self._predict_horizon(features_df, horizon)
                if pred is not None:
                    predictions[f'{horizon}h'] = pred
                    confidences[f'{horizon}h'] = conf
            
            if not predictions:
                result['status'] = 'prediction_failed'
                return result
            
            result['predictions'] = predictions
            result['model_confidence'] = confidences
            
            # Calculate prediction slope (trend)
            current_price = df['close'].iloc[-1]
            slopes = []
            
            for horizon, pred in predictions.items():
                hours = int(horizon.replace('h', ''))
                slope = (pred - current_price) / (current_price * hours)
                slopes.append(slope)
            
            result['prediction_slope'] = np.mean(slopes)
            
            # Determine direction
            if result['prediction_slope'] > 0.001:  # 0.1% per hour
                result['direction'] = 'long'
            elif result['prediction_slope'] < -0.001:
                result['direction'] = 'short'
            else:
                result['direction'] = None
            
            # Calculate timeframe alignment
            result['timeframe_alignment'] = self._calculate_timeframe_alignment(predictions, current_price)
            
            # Calculate overall confidence
            result['confidence'] = self._calculate_ml_confidence(
                confidences,
                result['timeframe_alignment'],
                abs(result['prediction_slope'])
            )
            
            # Get feature contributions
            result['feature_contributions'] = self._get_feature_contributions(features_df)
            
            # Add prediction metadata
            result['metadata'] = {
                'models_updated': {k: v.isoformat() if v else 'never' 
                                 for k, v in self.last_update.items()},
                'feature_count': len(features_df.columns),
                'sample_size': len(features_df)
            }
            
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            result['status'] = 'error'
            result['error'] = str(e)
        
        return result
    
    def _prepare_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Prepare features for ML models"""
        try:
            if len(df) < 100:
                return None
            
            features = pd.DataFrame(index=df.index)
            
            # Price returns
            features['returns_1h'] = df['close'].pct_change(1)
            features['returns_4h'] = df['close'].pct_change(4)
            features['returns_24h'] = df['close'].pct_change(24)
            
            # Volume features
            volume_sma = df['volume'].rolling(window=20).mean()
            features['volume_ratio'] = df['volume'] / volume_sma
            
            # Technical indicators
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema_12 = df['close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['close'].ewm(span=26, adjust=False).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9, adjust=False).mean()
            features['macd_signal'] = (macd - signal) / df['close']
            
            # Bollinger Bands position
            sma_20 = df['close'].rolling(window=20).mean()
            std_20 = df['close'].rolling(window=20).std()
            upper_band = sma_20 + (2 * std_20)
            lower_band = sma_20 - (2 * std_20)
            features['bb_position'] = (df['close'] - lower_band) / (upper_band - lower_band)
            
            # ATR ratio
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean()
            features['atr_ratio'] = atr / df['close']
            
            # Trend strength
            sma_50 = df['close'].rolling(window=50).mean()
            features['trend_strength'] = (df['close'] - sma_50) / sma_50
            
            # Time features (cyclical encoding)
            timestamps = pd.to_datetime(df.index)
            hours = timestamps.hour
            days = timestamps.dayofweek
            
            features['hour_sin'] = np.sin(2 * np.pi * hours / 24)
            features['hour_cos'] = np.cos(2 * np.pi * hours / 24)
            features['day_sin'] = np.sin(2 * np.pi * days / 7)
            features['day_cos'] = np.cos(2 * np.pi * days / 7)
            
            # Drop NaN values
            features = features.dropna()
            
            # Ensure we have the required features
            missing_features = set(self.model_params['features']) - set(features.columns)
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                return None
            
            return features[self.model_params['features']]
            
        except Exception as e:
            logger.error(f"Feature preparation error: {e}")
            return None
    
    def _predict_horizon(self, features: pd.DataFrame, horizon: int) -> Tuple[Optional[float], float]:
        """Make prediction for specific time horizon"""
        try:
            model_key = f'model_{horizon}h'
            scaler_key = f'scaler_{horizon}h'
            
            # Check if model exists
            if model_key not in self.models:
                logger.warning(f"No model for {horizon}h horizon")
                return None, 0.0
            
            model = self.models[model_key]
            scaler = self.scalers[scaler_key]
            
            # Scale features
            latest_features = features.iloc[-1:].values
            scaled_features = scaler.transform(latest_features)
            
            # Make prediction
            prediction = model.predict(scaled_features)[0]
            
            # Calculate confidence based on model performance
            confidence = self._calculate_model_confidence(model_key, features)
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Prediction error for {horizon}h: {e}")
            return None, 0.0
    
    def _calculate_timeframe_alignment(self, predictions: Dict[str, float], current_price: float) -> float:
        """Calculate how well different timeframe predictions align"""
        if len(predictions) < 2:
            return 0.5
        
        # Check if all predictions point in the same direction
        directions = []
        for pred in predictions.values():
            if pred > current_price * 1.001:
                directions.append(1)
            elif pred < current_price * 0.999:
                directions.append(-1)
            else:
                directions.append(0)
        
        # Calculate alignment
        if len(set(directions)) == 1 and directions[0] != 0:
            return 1.0  # Perfect alignment
        elif len(set(directions)) == 1:
            return 0.5  # All neutral
        else:
            # Partial alignment
            most_common = max(set(directions), key=directions.count)
            alignment = directions.count(most_common) / len(directions)
            return alignment
    
    def _calculate_ml_confidence(
        self,
        model_confidences: Dict[str, float],
        timeframe_alignment: float,
        prediction_strength: float
    ) -> float:
        """Calculate overall ML prediction confidence"""
        if not model_confidences:
            return 0.0
        
        # Average model confidence
        avg_model_conf = np.mean(list(model_confidences.values()))
        
        # Prediction strength factor (larger predicted moves = higher confidence)
        strength_factor = min(1.0, prediction_strength * 50)  # 2% move = 1.0
        
        # Combine factors
        confidence = (
            avg_model_conf * 0.5 +
            timeframe_alignment * 0.3 +
            strength_factor * 0.2
        )
        
        return min(1.0, confidence)
    
    def _calculate_model_confidence(self, model_key: str, features: pd.DataFrame) -> float:
        """Calculate confidence for a specific model"""
        # Base confidence on model performance
        if model_key in self.model_performance:
            perf = self.model_performance[model_key]
            # Use R² score as base confidence
            base_conf = max(0, perf.get('r2_score', 0.5))
        else:
            base_conf = 0.5
        
        # Adjust based on feature quality
        feature_quality = 1.0 - features.isna().sum().sum() / features.size
        
        # Adjust based on recent performance
        # In production, track recent prediction accuracy
        recency_factor = 0.8  # Placeholder
        
        confidence = base_conf * feature_quality * recency_factor
        
        return min(1.0, confidence)
    
    def _get_feature_contributions(self, features: pd.DataFrame) -> Dict[str, float]:
        """Get feature importance/contributions"""
        contributions = {}
        
        try:
            # Use the 1h model for feature importance
            model_key = 'model_1h'
            if model_key in self.models and hasattr(self.models[model_key], 'feature_importances_'):
                importances = self.models[model_key].feature_importances_
                feature_names = self.model_params['features']
                
                for name, importance in zip(feature_names, importances):
                    contributions[name] = float(importance)
                
                # Normalize
                total = sum(contributions.values())
                if total > 0:
                    contributions = {k: v/total for k, v in contributions.items()}
                    
        except Exception as e:
            logger.warning(f"Feature contribution error: {e}")
        
        return contributions
    
    def train_models(self, historical_data: pd.DataFrame, target_col: str = 'close'):
        """Train ML models on historical data"""
        try:
            logger.info("Training ML models...")
            
            # Prepare features
            features_df = self._prepare_features(historical_data)
            if features_df is None or len(features_df) < self.model_params['min_train_samples']:
                logger.warning("Insufficient data for training")
                return
            
            # Train model for each horizon
            for horizon in self.model_params['prediction_horizons']:
                logger.info(f"Training {horizon}h model...")
                
                # Prepare target (future price)
                target = historical_data[target_col].shift(-horizon)
                
                # Align features and target
                valid_idx = features_df.index.intersection(target.dropna().index)
                X = features_df.loc[valid_idx]
                y = target.loc[valid_idx]
                
                if len(X) < self.model_params['min_train_samples']:
                    logger.warning(f"Insufficient samples for {horizon}h model")
                    continue
                
                # Split data (simple time-based split)
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train ensemble model
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                )
                
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                train_score = model.score(X_train_scaled, y_train)
                test_score = model.score(X_test_scaled, y_test)
                
                logger.info(f"{horizon}h model - Train R²: {train_score:.3f}, Test R²: {test_score:.3f}")
                
                # Store model and scaler
                model_key = f'model_{horizon}h'
                scaler_key = f'scaler_{horizon}h'
                
                self.models[model_key] = model
                self.scalers[scaler_key] = scaler
                self.last_update[model_key] = datetime.now()
                self.model_performance[model_key] = {
                    'r2_score': test_score,
                    'train_score': train_score,
                    'samples': len(X)
                }
                
                # Save model
                self._save_model(model_key, model, scaler)
                
            logger.info("Model training completed")
            
        except Exception as e:
            logger.error(f"Model training error: {e}")
    
    def _save_model(self, model_key: str, model: Any, scaler: Any):
        """Save model and scaler to disk"""
        try:
            model_path = os.path.join(self.model_dir, f"{model_key}.pkl")
            scaler_path = os.path.join(self.model_dir, f"{model_key}_scaler.pkl")
            
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            
            logger.info(f"Saved {model_key}")
            
        except Exception as e:
            logger.error(f"Model save error: {e}")
    
    def _load_models(self):
        """Load existing models from disk"""
        try:
            for horizon in self.model_params['prediction_horizons']:
                model_key = f'model_{horizon}h'
                model_path = os.path.join(self.model_dir, f"{model_key}.pkl")
                scaler_path = os.path.join(self.model_dir, f"{model_key}_scaler.pkl")
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    self.models[model_key] = joblib.load(model_path)
                    self.scalers[f'scaler_{horizon}h'] = joblib.load(scaler_path)
                    self.last_update[model_key] = datetime.fromtimestamp(os.path.getmtime(model_path))
                    
                    logger.info(f"Loaded {model_key}")
                    
        except Exception as e:
            logger.error(f"Model load error: {e}")
    
    def is_healthy(self) -> bool:
        """Check if ML predictor is healthy"""
        # Check if we have at least one model
        return len(self.models) > 0