"""
BEAST Trading System - Pattern Analyzer
Integrates pattern_rules.py for comprehensive pattern detection
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging

from config.settings import config
from utils.logger import get_logger
import pattern_rules

logger = get_logger(__name__)

class PatternAnalyzer:
    """
    Comprehensive pattern analysis using pattern_rules.py
    NO RANDOM SIGNALS - Only detected patterns with confidence
    """
    
    def __init__(self, config):
        self.config = config
        self.pattern_config = config.trading.pattern_detection
        
        # Pattern categories and their functions from pattern_rules
        self.candlestick_patterns = {
            # Single candlestick patterns
            'hammer': pattern_rules.is_hammer,
            'inverted_hammer': pattern_rules.is_inverted_hammer,
            'bullish_marubozu': pattern_rules.is_bullish_marubozu,
            'bearish_marubozu': pattern_rules.is_bearish_marubozu,
            'doji': pattern_rules.is_doji,
            'spinning_top': pattern_rules.is_spinning_top,
            'hanging_man': pattern_rules.is_hanging_man,
            'shooting_star': pattern_rules.is_shooting_star,
            
            # Two-candlestick patterns
            'bullish_engulfing': pattern_rules.is_bullish_engulfing,
            'bearish_engulfing': pattern_rules.is_bearish_engulfing,
            'bullish_harami': pattern_rules.is_bullish_harami,
            'bearish_harami': pattern_rules.is_bearish_harami,
            'piercing_line': pattern_rules.is_piercing_line,
            'dark_cloud_cover': pattern_rules.is_dark_cloud_cover,
            
            # Three-candlestick patterns
            'morning_star': pattern_rules.is_morning_star,
            'evening_star': pattern_rules.is_evening_star,
            'three_white_soldiers': pattern_rules.is_three_white_soldiers,
            'three_black_crows': pattern_rules.is_three_black_crows,
            'three_inside_up': pattern_rules.is_three_inside_up,
            'three_inside_down': pattern_rules.is_three_inside_down,
            'three_outside_up': pattern_rules.is_three_outside_up,
            'three_outside_down': pattern_rules.is_three_outside_down,
            'abandoned_baby_bullish': pattern_rules.is_abandoned_baby_bullish,
            'abandoned_baby_bearish': pattern_rules.is_abandoned_baby_bearish,
            
            # Five-candlestick patterns
            'rising_three_methods': pattern_rules.is_rising_three_methods,
            'falling_three_methods': pattern_rules.is_falling_three_methods
        }
        
        self.chart_patterns = {
            # Classic chart patterns
            'head_and_shoulders': pattern_rules.is_head_and_shoulders,
            'inverse_head_and_shoulders': pattern_rules.is_inverse_head_and_shoulders,
            'double_top': pattern_rules.is_double_top,
            'double_bottom': pattern_rules.is_double_bottom,
            
            # Channel patterns
            'ascending_channel': pattern_rules.is_ascending_channel,
            'descending_channel': pattern_rules.is_descending_channel,
            'horizontal_channel': pattern_rules.is_horizontal_channel,
            
            # Reversal patterns
            'rounding_top': pattern_rules.is_rounding_top,
            'rounding_bottom': pattern_rules.is_rounding_bottom,
            'v_bottom': pattern_rules.is_v_bottom,
            'inverted_v_top': pattern_rules.is_inverted_v_top,
            'key_reversal': pattern_rules.is_key_reversal,
            'island_reversal': pattern_rules.is_island_reversal,
            
            # Diamond patterns
            'diamond_top': pattern_rules.is_diamond_top,
            'diamond_bottom': pattern_rules.is_diamond_bottom,
            
            # Other chart patterns
            'coil_pattern': pattern_rules.is_coil_pattern,
            'bump_and_run': pattern_rules.is_bump_and_run,
            'scallop_pattern': pattern_rules.is_scallop_pattern,
            'ascending_staircase': pattern_rules.is_ascending_staircase,
            'descending_staircase': pattern_rules.is_descending_staircase
        }
        
        self.harmonic_patterns = {
            'gartley': pattern_rules.is_gartley_pattern,
            'bat': pattern_rules.is_bat_pattern,
            'butterfly': pattern_rules.is_butterfly_pattern,
            'crab': pattern_rules.is_crab_pattern,
            'shark': pattern_rules.is_shark_pattern,
            'cypher': pattern_rules.is_cypher_pattern,
            'three_drives': pattern_rules.is_three_drives_pattern
        }
        
        self.elliott_wave_patterns = {
            'impulse_wave': pattern_rules.is_impulse_wave,
            'corrective_wave': pattern_rules.is_corrective_wave
        }
        
        # Pattern strength and reliability scores
        self.pattern_reliability = self._initialize_pattern_reliability()
        
        # Performance tracking
        self.pattern_performance = {}
        
        logger.info(f"PatternAnalyzer initialized with {self._count_patterns()} patterns")
    
    def _count_patterns(self) -> int:
        """Count total number of patterns"""
        return (len(self.candlestick_patterns) + 
                len(self.chart_patterns) + 
                len(self.harmonic_patterns) + 
                len(self.elliott_wave_patterns))
    
    def _initialize_pattern_reliability(self) -> Dict[str, float]:
        """Initialize pattern reliability scores based on historical performance"""
        reliability = {
            # High reliability candlestick patterns
            'bullish_engulfing': 0.75,
            'bearish_engulfing': 0.75,
            'morning_star': 0.80,
            'evening_star': 0.80,
            'three_white_soldiers': 0.78,
            'three_black_crows': 0.78,
            
            # Medium reliability candlestick patterns
            'hammer': 0.65,
            'inverted_hammer': 0.65,
            'shooting_star': 0.65,
            'hanging_man': 0.65,
            'doji': 0.60,
            'piercing_line': 0.70,
            'dark_cloud_cover': 0.70,
            
            # High reliability chart patterns
            'head_and_shoulders': 0.85,
            'inverse_head_and_shoulders': 0.85,
            'double_top': 0.80,
            'double_bottom': 0.80,
            
            # Medium reliability chart patterns
            'ascending_channel': 0.70,
            'descending_channel': 0.70,
            'horizontal_channel': 0.65,
            'v_bottom': 0.75,
            'inverted_v_top': 0.75,
            
            # Harmonic patterns (generally high reliability)
            'gartley': 0.78,
            'bat': 0.75,
            'butterfly': 0.73,
            'crab': 0.70,
            
            # Elliott waves
            'impulse_wave': 0.72,
            'corrective_wave': 0.68
        }
        
        # Set default reliability for unspecified patterns
        default_reliability = 0.65
        all_patterns = (list(self.candlestick_patterns.keys()) + 
                       list(self.chart_patterns.keys()) + 
                       list(self.harmonic_patterns.keys()) + 
                       list(self.elliott_wave_patterns.keys()))
        
        for pattern in all_patterns:
            if pattern not in reliability:
                reliability[pattern] = default_reliability
        
        return reliability
    
    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze price data for patterns
        Returns detected patterns with confidence scores
        """
        result = {
            'status': 'analyzed',
            'patterns': [],
            'confidence': 0.0,
            'direction': None,
            'signals': {},
            'pattern_summary': {}
        }
        
        try:
            # Validate input data
            if 'price_data' not in data or data['price_data'] is None:
                result['status'] = 'no_data'
                return result
            
            df = data['price_data']
            
            # Ensure proper column names for pattern_rules functions
            df_normalized = self._normalize_dataframe(df)
            
            # Detect all pattern types
            detected_patterns = []
            
            # 1. Candlestick patterns
            candlestick_detected = self._detect_candlestick_patterns(df_normalized)
            detected_patterns.extend(candlestick_detected)
            
            # 2. Chart patterns (need more data)
            if len(df_normalized) >= self.pattern_config['chart_pattern_window']:
                chart_detected = self._detect_chart_patterns(df_normalized)
                detected_patterns.extend(chart_detected)
            
            # 3. Harmonic patterns
            if len(df_normalized) >= 20:
                harmonic_detected = self._detect_harmonic_patterns(df_normalized)
                detected_patterns.extend(harmonic_detected)
            
            # 4. Elliott Wave patterns
            if len(df_normalized) >= 10:
                elliott_detected = self._detect_elliott_patterns(df_normalized)
                detected_patterns.extend(elliott_detected)
            
            # Process detected patterns
            if detected_patterns:
                result['patterns'] = detected_patterns
                result['confidence'] = self._calculate_pattern_confidence(detected_patterns)
                result['direction'] = self._determine_pattern_direction(detected_patterns)
                result['signals'] = self._generate_pattern_signals(detected_patterns)
                result['pattern_summary'] = self._create_pattern_summary(detected_patterns)
            else:
                result['status'] = 'no_patterns_found'
                result['confidence'] = 0.0
                
            # Add pattern context
            result['context'] = self._analyze_pattern_context(df_normalized, detected_patterns)
            
        except Exception as e:
            logger.error(f"Pattern analysis error: {e}")
            result['status'] = 'error'
            result['error'] = str(e)
        
        return result
    
    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize dataframe columns for pattern_rules functions"""
        # Create a copy to avoid modifying original
        normalized = df.copy()
        
        # Ensure required columns with proper capitalization
        column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in normalized.columns and new_col not in normalized.columns:
                normalized[new_col] = normalized[old_col]
        
        return normalized
    
    def _detect_candlestick_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect candlestick patterns"""
        detected = []
        
        for pattern_name, pattern_func in self.candlestick_patterns.items():
            try:
                # Get required window size for pattern
                window_size = self._get_pattern_window(pattern_name)
                
                if len(df) >= window_size:
                    # Check if pattern is present
                    if pattern_func(df):
                        detected.append({
                            'type': 'candlestick',
                            'name': pattern_name,
                            'reliability': self.pattern_reliability.get(pattern_name, 0.65),
                            'window': window_size,
                            'timestamp': df.index[-1],
                            'price': df['Close'].iloc[-1]
                        })
                        
            except Exception as e:
                logger.debug(f"Error detecting {pattern_name}: {e}")
                continue
        
        return detected
    
    def _detect_chart_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect chart patterns"""
        detected = []
        
        for pattern_name, pattern_func in self.chart_patterns.items():
            try:
                if pattern_func(df):
                    detected.append({
                        'type': 'chart',
                        'name': pattern_name,
                        'reliability': self.pattern_reliability.get(pattern_name, 0.70),
                        'window': len(df),
                        'timestamp': df.index[-1],
                        'price': df['Close'].iloc[-1]
                    })
                    
            except Exception as e:
                logger.debug(f"Error detecting {pattern_name}: {e}")
                continue
        
        return detected
    
    def _detect_harmonic_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect harmonic patterns"""
        detected = []
        
        for pattern_name, pattern_func in self.harmonic_patterns.items():
            try:
                if pattern_func(df):
                    detected.append({
                        'type': 'harmonic',
                        'name': pattern_name,
                        'reliability': self.pattern_reliability.get(pattern_name, 0.75),
                        'window': len(df),
                        'timestamp': df.index[-1],
                        'price': df['Close'].iloc[-1]
                    })
                    
            except Exception as e:
                logger.debug(f"Error detecting {pattern_name}: {e}")
                continue
        
        return detected
    
    def _detect_elliott_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect Elliott Wave patterns"""
        detected = []
        
        for pattern_name, pattern_func in self.elliott_wave_patterns.items():
            try:
                if pattern_func(df):
                    detected.append({
                        'type': 'elliott',
                        'name': pattern_name,
                        'reliability': self.pattern_reliability.get(pattern_name, 0.70),
                        'window': len(df),
                        'timestamp': df.index[-1],
                        'price': df['Close'].iloc[-1]
                    })
                    
            except Exception as e:
                logger.debug(f"Error detecting {pattern_name}: {e}")
                continue
        
        return detected
    
    def _get_pattern_window(self, pattern_name: str) -> int:
        """Get required window size for pattern"""
        # Candlestick pattern windows
        single_candle = ['hammer', 'inverted_hammer', 'doji', 'spinning_top', 
                        'bullish_marubozu', 'bearish_marubozu']
        two_candle = ['bullish_engulfing', 'bearish_engulfing', 'bullish_harami', 
                     'bearish_harami', 'piercing_line', 'dark_cloud_cover']
        three_candle = ['morning_star', 'evening_star', 'three_white_soldiers', 
                       'three_black_crows', 'three_inside_up', 'three_inside_down']
        five_candle = ['rising_three_methods', 'falling_three_methods']
        
        if pattern_name in single_candle:
            return 1
        elif pattern_name in two_candle:
            return 2
        elif pattern_name in three_candle:
            return 3
        elif pattern_name in five_candle:
            return 5
        elif 'hanging_man' in pattern_name or 'shooting_star' in pattern_name:
            return 6  # Need context for trend
        else:
            return self.pattern_config['candlestick_window']
    
    def _calculate_pattern_confidence(self, patterns: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence from detected patterns"""
        if not patterns:
            return 0.0
        
        # Weight by pattern reliability and type
        type_weights = {
            'candlestick': 0.25,
            'chart': 0.35,
            'harmonic': 0.30,
            'elliott': 0.10
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for pattern in patterns:
            pattern_type = pattern['type']
            reliability = pattern['reliability']
            weight = type_weights.get(pattern_type, 0.2)
            
            weighted_sum += reliability * weight
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            base_confidence = weighted_sum / total_weight
        else:
            base_confidence = 0.0
        
        # Boost confidence if multiple patterns agree
        if len(patterns) >= 3:
            base_confidence *= 1.2
        elif len(patterns) >= 2:
            base_confidence *= 1.1
        
        # Cap at 1.0
        return min(1.0, base_confidence)
    
    def _determine_pattern_direction(self, patterns: List[Dict[str, Any]]) -> Optional[str]:
        """Determine overall direction from patterns"""
        bullish_patterns = [
            'hammer', 'inverted_hammer', 'bullish_marubozu', 'bullish_engulfing',
            'bullish_harami', 'piercing_line', 'morning_star', 'three_white_soldiers',
            'three_inside_up', 'three_outside_up', 'abandoned_baby_bullish',
            'rising_three_methods', 'inverse_head_and_shoulders', 'double_bottom',
            'ascending_channel', 'rounding_bottom', 'v_bottom', 'gartley', 'bat',
            'impulse_wave'
        ]
        
        bearish_patterns = [
            'hanging_man', 'shooting_star', 'bearish_marubozu', 'bearish_engulfing',
            'bearish_harami', 'dark_cloud_cover', 'evening_star', 'three_black_crows',
            'three_inside_down', 'three_outside_down', 'abandoned_baby_bearish',
            'falling_three_methods', 'head_and_shoulders', 'double_top',
            'descending_channel', 'rounding_top', 'inverted_v_top', 'corrective_wave'
        ]
        
        bullish_count = sum(1 for p in patterns if p['name'] in bullish_patterns)
        bearish_count = sum(1 for p in patterns if p['name'] in bearish_patterns)
        
        if bullish_count > bearish_count:
            return 'long'
        elif bearish_count > bullish_count:
            return 'short'
        else:
            return None
    
    def _generate_pattern_signals(self, patterns: List[Dict[str, Any]]) -> Dict[str, float]:
        """Generate trading signals from patterns"""
        signals = {}
        
        for pattern in patterns:
            signal_name = f"{pattern['type']}_{pattern['name']}"
            signals[signal_name] = pattern['reliability']
        
        # Add aggregate signals
        if len(patterns) >= 2:
            signals['multiple_patterns'] = 0.8
        
        # Add pattern type signals
        pattern_types = set(p['type'] for p in patterns)
        if len(pattern_types) >= 2:
            signals['multi_type_confirmation'] = 0.9
        
        return signals
    
    def _create_pattern_summary(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary of detected patterns"""
        summary = {
            'total_patterns': len(patterns),
            'by_type': {},
            'strongest_pattern': None,
            'weakest_pattern': None,
            'avg_reliability': 0.0
        }
        
        # Count by type
        for pattern in patterns:
            pattern_type = pattern['type']
            if pattern_type not in summary['by_type']:
                summary['by_type'][pattern_type] = []
            summary['by_type'][pattern_type].append(pattern['name'])
        
        # Find strongest and weakest
        if patterns:
            sorted_patterns = sorted(patterns, key=lambda x: x['reliability'], reverse=True)
            summary['strongest_pattern'] = sorted_patterns[0]['name']
            summary['weakest_pattern'] = sorted_patterns[-1]['name']
            summary['avg_reliability'] = np.mean([p['reliability'] for p in patterns])
        
        return summary
    
    def _analyze_pattern_context(self, df: pd.DataFrame, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the context around detected patterns"""
        context = {
            'trend': self._determine_trend(df),
            'volatility': self._calculate_volatility(df),
            'volume_profile': self._analyze_volume(df),
            'support_resistance': self._find_support_resistance(df)
        }
        
        return context
    
    def _determine_trend(self, df: pd.DataFrame) -> str:
        """Determine overall trend"""
        if len(df) < 20:
            return 'unknown'
        
        # Use simple moving average comparison
        sma_short = df['Close'].rolling(window=10).mean().iloc[-1]
        sma_long = df['Close'].rolling(window=20).mean().iloc[-1]
        
        if sma_short > sma_long * 1.01:
            return 'uptrend'
        elif sma_short < sma_long * 0.99:
            return 'downtrend'
        else:
            return 'sideways'
    
    def _calculate_volatility(self, df: pd.DataFrame) -> str:
        """Calculate current volatility state"""
        returns = df['Close'].pct_change().dropna()
        current_vol = returns.rolling(window=10).std().iloc[-1]
        avg_vol = returns.rolling(window=50).std().mean()
        
        if current_vol > avg_vol * 1.5:
            return 'high'
        elif current_vol < avg_vol * 0.5:
            return 'low'
        else:
            return 'normal'
    
    def _analyze_volume(self, df: pd.DataFrame) -> str:
        """Analyze volume profile"""
        if 'Volume' not in df.columns:
            return 'unknown'
        
        current_vol = df['Volume'].iloc[-5:].mean()
        avg_vol = df['Volume'].rolling(window=20).mean().iloc[-1]
        
        if current_vol > avg_vol * 1.5:
            return 'high_volume'
        elif current_vol < avg_vol * 0.5:
            return 'low_volume'
        else:
            return 'normal_volume'
    
    def _find_support_resistance(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """Find key support and resistance levels"""
        highs = df['High'].values
        lows = df['Low'].values
        
        # Simple peak/trough based S/R
        peaks, troughs = pattern_rules.find_peaks_and_troughs(df['Close'].values)
        
        resistance_levels = []
        support_levels = []
        
        if peaks:
            # Get last 3 peaks as resistance
            for idx in peaks[-3:]:
                resistance_levels.append(highs[idx])
        
        if troughs:
            # Get last 3 troughs as support
            for idx in troughs[-3:]:
                support_levels.append(lows[idx])
        
        return {
            'resistance': sorted(set(resistance_levels), reverse=True),
            'support': sorted(set(support_levels))
        }
    
    def update_pattern_performance(self, pattern_name: str, success: bool):
        """Update pattern performance tracking"""
        if pattern_name not in self.pattern_performance:
            self.pattern_performance[pattern_name] = {
                'occurrences': 0,
                'successes': 0,
                'success_rate': 0.0
            }
        
        stats = self.pattern_performance[pattern_name]
        stats['occurrences'] += 1
        if success:
            stats['successes'] += 1
        stats['success_rate'] = stats['successes'] / stats['occurrences']
        
        # Update reliability score based on performance
        if stats['occurrences'] >= 10:
            # Blend historical reliability with actual performance
            historical = self.pattern_reliability.get(pattern_name, 0.65)
            actual = stats['success_rate']
            self.pattern_reliability[pattern_name] = 0.7 * historical + 0.3 * actual
    
    def is_healthy(self) -> bool:
        """Check if pattern analyzer is healthy"""
        return True  # Pattern analyzer is stateless and always healthy