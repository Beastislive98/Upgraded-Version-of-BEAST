"""
BEAST Trading System - Technical Analyzer
Comprehensive technical analysis with multiple indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import talib
from datetime import datetime
import logging

from config.settings import config
from utils.logger import get_logger

logger = get_logger(__name__)

class TechnicalAnalyzer:
    """
    Technical analysis module providing indicators and signals
    NO RANDOM COMPONENTS - Pure calculation based analysis
    """
    
    def __init__(self, config):
        self.config = config
        
        # Indicator settings
        self.indicator_params = {
            'rsi': {'period': 14, 'overbought': 70, 'oversold': 30},
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'bollinger': {'period': 20, 'std_dev': 2},
            'stochastic': {'k_period': 14, 'd_period': 3, 'overbought': 80, 'oversold': 20},
            'atr': {'period': 14},
            'adx': {'period': 14, 'trend_strength': 25},
            'ema': {'periods': [9, 21, 50, 200]},
            'sma': {'periods': [20, 50, 100, 200]},
            'volume': {'period': 20}
        }
        
        # Signal thresholds
        self.signal_thresholds = {
            'strong_bullish': 0.8,
            'bullish': 0.6,
            'neutral': 0.4,
            'bearish': 0.2,
            'strong_bearish': 0.0
        }
        
        logger.info("TechnicalAnalyzer initialized")
    
    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive technical analysis
        """
        result = {
            'status': 'analyzed',
            'indicators': {},
            'signals': {},
            'confidence': 0.0,
            'direction': None,
            'trend': None,
            'strength': 0.0,
            'volume_state': 'normal'
        }
        
        try:
            # Validate input data
            if 'price_data' not in data or data['price_data'] is None:
                result['status'] = 'no_data'
                return result
            
            df = data['price_data']
            
            # Calculate all indicators
            indicators = self._calculate_indicators(df)
            result['indicators'] = indicators
            
            # Generate trading signals
            signals = self._generate_signals(indicators, df)
            result['signals'] = signals
            
            # Determine trend
            trend_analysis = self._analyze_trend(df, indicators)
            result.update(trend_analysis)
            
            # Analyze volume
            volume_analysis = self._analyze_volume(df)
            result['volume_state'] = volume_analysis
            
            # Calculate overall confidence
            confidence = self._calculate_confidence(signals, trend_analysis, volume_analysis)
            result['confidence'] = confidence
            
            # Multi-timeframe analysis if available
            if hasattr(df, 'attrs') and 'timeframes' in df.attrs:
                mtf_analysis = self._multi_timeframe_analysis(df.attrs['timeframes'])
                result['multi_timeframe'] = mtf_analysis
            
        except Exception as e:
            logger.error(f"Technical analysis error: {e}")
            result['status'] = 'error'
            result['error'] = str(e)
        
        return result
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all technical indicators"""
        indicators = {}
        
        # Ensure we have enough data
        if len(df) < 200:  # Need at least 200 periods for some indicators
            logger.warning("Insufficient data for all indicators")
        
        # Price data
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # Momentum Indicators
        try:
            # RSI
            indicators['RSI_14'] = talib.RSI(close, timeperiod=self.indicator_params['rsi']['period'])[-1]
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(
                close, 
                fastperiod=self.indicator_params['macd']['fast'],
                slowperiod=self.indicator_params['macd']['slow'],
                signalperiod=self.indicator_params['macd']['signal']
            )
            indicators['MACD'] = macd[-1] if len(macd) > 0 else 0
            indicators['MACD_signal'] = macd_signal[-1] if len(macd_signal) > 0 else 0
            indicators['MACD_histogram'] = macd_hist[-1] if len(macd_hist) > 0 else 0
            
            # Stochastic
            slowk, slowd = talib.STOCH(
                high, low, close,
                fastk_period=self.indicator_params['stochastic']['k_period'],
                slowk_period=3,
                slowd_period=self.indicator_params['stochastic']['d_period']
            )
            indicators['STOCH_K'] = slowk[-1] if len(slowk) > 0 else 50
            indicators['STOCH_D'] = slowd[-1] if len(slowd) > 0 else 50
            
            # ADX (Average Directional Index)
            indicators['ADX'] = talib.ADX(high, low, close, timeperiod=self.indicator_params['adx']['period'])[-1]
            
            # CCI (Commodity Channel Index)
            indicators['CCI'] = talib.CCI(high, low, close, timeperiod=20)[-1]
            
        except Exception as e:
            logger.warning(f"Error calculating momentum indicators: {e}")
        
        # Trend Indicators
        try:
            # Moving Averages
            for period in self.indicator_params['sma']['periods']:
                if len(close) >= period:
                    indicators[f'SMA_{period}'] = talib.SMA(close, timeperiod=period)[-1]
            
            for period in self.indicator_params['ema']['periods']:
                if len(close) >= period:
                    indicators[f'EMA_{period}'] = talib.EMA(close, timeperiod=period)[-1]
            
            # TEMA (Triple Exponential Moving Average)
            if len(close) >= 30:
                indicators['TEMA_30'] = talib.TEMA(close, timeperiod=30)[-1]
            
        except Exception as e:
            logger.warning(f"Error calculating trend indicators: {e}")
        
        # Volatility Indicators
        try:
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(
                close,
                timeperiod=self.indicator_params['bollinger']['period'],
                nbdevup=self.indicator_params['bollinger']['std_dev'],
                nbdevdn=self.indicator_params['bollinger']['std_dev']
            )
            indicators['BB_upper'] = upper[-1] if len(upper) > 0 else 0
            indicators['BB_middle'] = middle[-1] if len(middle) > 0 else 0
            indicators['BB_lower'] = lower[-1] if len(lower) > 0 else 0
            indicators['BB_width'] = (upper[-1] - lower[-1]) / middle[-1] if middle[-1] > 0 else 0
            
            # ATR (Average True Range)
            indicators['ATR_14'] = talib.ATR(high, low, close, timeperiod=self.indicator_params['atr']['period'])[-1]
            
            # Historical Volatility
            indicators['volatility'] = df['returns'].rolling(window=20).std().iloc[-1] if 'returns' in df else 0
            
        except Exception as e:
            logger.warning(f"Error calculating volatility indicators: {e}")
        
        # Volume Indicators
        try:
            # OBV (On Balance Volume)
            indicators['OBV'] = talib.OBV(close, volume)[-1]
            
            # Volume SMA
            indicators['volume_sma'] = talib.SMA(volume, timeperiod=self.indicator_params['volume']['period'])[-1]
            indicators['volume_ratio'] = volume[-1] / indicators['volume_sma'] if indicators['volume_sma'] > 0 else 1
            
            # MFI (Money Flow Index)
            indicators['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)[-1]
            
        except Exception as e:
            logger.warning(f"Error calculating volume indicators: {e}")
        
        # Pattern Recognition Indicators
        try:
            # Parabolic SAR
            indicators['SAR'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)[-1]
            
            # Heikin-Ashi values
            ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
            indicators['HA_trend'] = 'bullish' if ha_close.iloc[-1] > ha_close.iloc[-2] else 'bearish'
            
        except Exception as e:
            logger.warning(f"Error calculating pattern indicators: {e}")
        
        # Additional custom indicators
        indicators.update(self._calculate_custom_indicators(df))
        
        return indicators
    
    def _calculate_custom_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate custom technical indicators"""
        custom = {}
        
        try:
            # Price position within range
            high_low_range = df['high'].rolling(window=20).max() - df['low'].rolling(window=20).min()
            price_position = (df['close'] - df['low'].rolling(window=20).min()) / high_low_range
            custom['price_position'] = price_position.iloc[-1] if not price_position.empty else 0.5
            
            # Trend strength (based on ADX and moving average alignment)
            if all(col in df.columns for col in ['SMA_20', 'SMA_50', 'SMA_200']):
                ma_alignment = (
                    (df['SMA_20'] > df['SMA_50']).astype(int) +
                    (df['SMA_50'] > df['SMA_200']).astype(int)
                ) / 2
                custom['ma_alignment'] = ma_alignment.iloc[-1] if not ma_alignment.empty else 0.5
            
            # Support/Resistance levels
            recent_high = df['high'].rolling(window=20).max().iloc[-1]
            recent_low = df['low'].rolling(window=20).min().iloc[-1]
            current_price = df['close'].iloc[-1]
            
            custom['distance_to_resistance'] = (recent_high - current_price) / current_price
            custom['distance_to_support'] = (current_price - recent_low) / current_price
            
        except Exception as e:
            logger.warning(f"Error calculating custom indicators: {e}")
        
        return custom
    
    def _generate_signals(self, indicators: Dict[str, Any], df: pd.DataFrame) -> Dict[str, float]:
        """Generate trading signals from indicators"""
        signals = {}
        
        # RSI signals
        rsi = indicators.get('RSI_14', 50)
        if rsi > self.indicator_params['rsi']['overbought']:
            signals['rsi_overbought'] = 1.0
        elif rsi < self.indicator_params['rsi']['oversold']:
            signals['rsi_oversold'] = 1.0
        else:
            signals['rsi_neutral'] = (rsi - 30) / 40  # Normalize to 0-1
        
        # MACD signals
        macd = indicators.get('MACD', 0)
        macd_signal = indicators.get('MACD_signal', 0)
        if macd > macd_signal:
            signals['macd_bullish'] = min(1.0, abs(macd - macd_signal) * 100)
        else:
            signals['macd_bearish'] = min(1.0, abs(macd - macd_signal) * 100)
        
        # Bollinger Band signals
        current_price = df['close'].iloc[-1]
        bb_upper = indicators.get('BB_upper', current_price)
        bb_lower = indicators.get('BB_lower', current_price)
        bb_middle = indicators.get('BB_middle', current_price)
        
        if current_price > bb_upper:
            signals['bb_overbought'] = 1.0
        elif current_price < bb_lower:
            signals['bb_oversold'] = 1.0
        else:
            # Position within bands
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5
            signals['bb_position'] = bb_position
        
        # Moving average signals
        ma_signals = self._generate_ma_signals(indicators, current_price)
        signals.update(ma_signals)
        
        # Volume signals
        volume_ratio = indicators.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:
            signals['high_volume'] = min(1.0, volume_ratio / 2)
        elif volume_ratio < 0.5:
            signals['low_volume'] = 1.0 - volume_ratio
        
        # ADX trend strength
        adx = indicators.get('ADX', 0)
        if adx > self.indicator_params['adx']['trend_strength']:
            signals['strong_trend'] = min(1.0, adx / 50)
        else:
            signals['weak_trend'] = 1.0 - (adx / self.indicator_params['adx']['trend_strength'])
        
        # MFI (Money Flow) signals
        mfi = indicators.get('MFI', 50)
        if mfi > 80:
            signals['mfi_overbought'] = 1.0
        elif mfi < 20:
            signals['mfi_oversold'] = 1.0
        
        return signals
    
    def _generate_ma_signals(self, indicators: Dict[str, Any], current_price: float) -> Dict[str, float]:
        """Generate moving average based signals"""
        ma_signals = {}
        
        # Check golden cross / death cross
        sma_50 = indicators.get('SMA_50', 0)
        sma_200 = indicators.get('SMA_200', 0)
        
        if sma_50 > 0 and sma_200 > 0:
            if sma_50 > sma_200:
                ma_signals['golden_cross'] = min(1.0, (sma_50 - sma_200) / sma_200)
            else:
                ma_signals['death_cross'] = min(1.0, (sma_200 - sma_50) / sma_50)
        
        # Price vs MA signals
        ma_periods = [20, 50, 200]
        above_ma_count = 0
        
        for period in ma_periods:
            ma_key = f'SMA_{period}'
            if ma_key in indicators and indicators[ma_key] > 0:
                if current_price > indicators[ma_key]:
                    above_ma_count += 1
        
        ma_signals['ma_support'] = above_ma_count / len(ma_periods)
        
        return ma_signals
    
    def _analyze_trend(self, df: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market trend"""
        trend_analysis = {
            'trend': 'neutral',
            'trend_strength': 0.0,
            'direction': None
        }
        
        try:
            # Use multiple methods to determine trend
            
            # Method 1: Moving average alignment
            ma_trend = self._ma_trend_analysis(indicators)
            
            # Method 2: Price action trend
            price_trend = self._price_action_trend(df)
            
            # Method 3: ADX-based trend
            adx_trend = self._adx_trend_analysis(indicators)
            
            # Combine methods
            trend_scores = {
                'bullish': 0,
                'bearish': 0,
                'neutral': 0
            }
            
            # Weight each method
            for trend, weight in [(ma_trend, 0.4), (price_trend, 0.4), (adx_trend, 0.2)]:
                if trend['direction'] == 'up':
                    trend_scores['bullish'] += weight * trend['strength']
                elif trend['direction'] == 'down':
                    trend_scores['bearish'] += weight * trend['strength']
                else:
                    trend_scores['neutral'] += weight
            
            # Determine overall trend
            if trend_scores['bullish'] > 0.6:
                trend_analysis['trend'] = 'uptrend'
                trend_analysis['direction'] = 'long'
                trend_analysis['trend_strength'] = trend_scores['bullish']
            elif trend_scores['bearish'] > 0.6:
                trend_analysis['trend'] = 'downtrend'
                trend_analysis['direction'] = 'short'
                trend_analysis['trend_strength'] = trend_scores['bearish']
            else:
                trend_analysis['trend'] = 'sideways'
                trend_analysis['trend_strength'] = trend_scores['neutral']
            
        except Exception as e:
            logger.warning(f"Error analyzing trend: {e}")
        
        return trend_analysis
    
    def _ma_trend_analysis(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trend using moving averages"""
        ma_periods = [20, 50, 200]
        available_mas = []
        
        for period in ma_periods:
            ma_key = f'SMA_{period}'
            if ma_key in indicators:
                available_mas.append((period, indicators[ma_key]))
        
        if len(available_mas) < 2:
            return {'direction': 'neutral', 'strength': 0.0}
        
        # Check alignment
        aligned_up = all(available_mas[i][1] > available_mas[i+1][1] 
                        for i in range(len(available_mas)-1))
        aligned_down = all(available_mas[i][1] < available_mas[i+1][1] 
                          for i in range(len(available_mas)-1))
        
        if aligned_up:
            return {'direction': 'up', 'strength': 0.8}
        elif aligned_down:
            return {'direction': 'down', 'strength': 0.8}
        else:
            return {'direction': 'neutral', 'strength': 0.3}
    
    def _price_action_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trend using price action"""
        if len(df) < 20:
            return {'direction': 'neutral', 'strength': 0.0}
        
        # Calculate trend using linear regression
        prices = df['close'].values[-20:]
        x = np.arange(len(prices))
        
        # Fit linear trend
        coeffs = np.polyfit(x, prices, 1)
        slope = coeffs[0]
        
        # Normalize slope
        price_range = prices.max() - prices.min()
        normalized_slope = slope / price_range if price_range > 0 else 0
        
        if normalized_slope > 0.02:
            return {'direction': 'up', 'strength': min(1.0, normalized_slope * 10)}
        elif normalized_slope < -0.02:
            return {'direction': 'down', 'strength': min(1.0, abs(normalized_slope) * 10)}
        else:
            return {'direction': 'neutral', 'strength': 0.2}
    
    def _adx_trend_analysis(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trend strength using ADX"""
        adx = indicators.get('ADX', 0)
        
        if adx > 40:
            return {'direction': 'strong', 'strength': 1.0}
        elif adx > 25:
            return {'direction': 'moderate', 'strength': 0.6}
        else:
            return {'direction': 'neutral', 'strength': 0.2}
    
    def _analyze_volume(self, df: pd.DataFrame) -> str:
        """Analyze volume patterns"""
        if 'volume' not in df.columns:
            return 'unknown'
        
        current_volume = df['volume'].iloc[-5:].mean()
        avg_volume = df['volume'].rolling(window=20).mean().iloc[-20:].mean()
        
        if current_volume > avg_volume * 1.5:
            return 'high'
        elif current_volume < avg_volume * 0.5:
            return 'low'
        else:
            return 'normal'
    
    def _calculate_confidence(
        self, 
        signals: Dict[str, float],
        trend_analysis: Dict[str, Any],
        volume_state: str
    ) -> float:
        """Calculate overall technical analysis confidence"""
        confidence = 0.5  # Base confidence
        
        # Trend strength contribution
        confidence += trend_analysis['trend_strength'] * 0.2
        
        # Signal agreement contribution
        bullish_signals = sum(1 for k, v in signals.items() 
                            if ('bullish' in k or 'oversold' in k) and v > 0.5)
        bearish_signals = sum(1 for k, v in signals.items() 
                            if ('bearish' in k or 'overbought' in k) and v > 0.5)
        
        signal_agreement = abs(bullish_signals - bearish_signals) / max(len(signals), 1)
        confidence += signal_agreement * 0.2
        
        # Volume confirmation
        if volume_state == 'high' and trend_analysis['trend'] != 'sideways':
            confidence += 0.1
        
        # Strong trend bonus
        if 'strong_trend' in signals and signals['strong_trend'] > 0.7:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _multi_timeframe_analysis(self, timeframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Perform multi-timeframe analysis"""
        mtf_results = {
            'alignment': 0.0,
            'timeframe_trends': {},
            'confluence_zones': []
        }
        
        try:
            trends = {}
            
            for tf, df in timeframes.items():
                if len(df) >= 20:
                    # Simple trend detection for each timeframe
                    sma20 = df['close'].rolling(window=20).mean()
                    if not sma20.empty:
                        current_price = df['close'].iloc[-1]
                        sma_value = sma20.iloc[-1]
                        
                        if current_price > sma_value * 1.01:
                            trends[tf] = 'bullish'
                        elif current_price < sma_value * 0.99:
                            trends[tf] = 'bearish'
                        else:
                            trends[tf] = 'neutral'
            
            mtf_results['timeframe_trends'] = trends
            
            # Calculate alignment
            if trends:
                trend_values = list(trends.values())
                bullish_count = trend_values.count('bullish')
                bearish_count = trend_values.count('bearish')
                
                if bullish_count == len(trend_values):
                    mtf_results['alignment'] = 1.0
                elif bearish_count == len(trend_values):
                    mtf_results['alignment'] = 1.0
                else:
                    mtf_results['alignment'] = max(bullish_count, bearish_count) / len(trend_values)
            
        except Exception as e:
            logger.warning(f"Error in multi-timeframe analysis: {e}")
        
        return mtf_results
    
    def is_healthy(self) -> bool:
        """Check if technical analyzer is healthy"""
        return True  # Technical analyzer is stateless and always healthy