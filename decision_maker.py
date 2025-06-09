"""
BEAST Trading System - Decision Maker
Implements strict confidence-based trading decisions with NO random trades
"""

import logging
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime, timezone
import numpy as np
from dataclasses import dataclass
import json

from config.settings import config, CONFIDENCE_THRESHOLD, MODULE_WEIGHTS, REQUIRED_MODULES
from utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class DecisionRecord:
    """Record of a trading decision for audit trail"""
    timestamp: datetime
    symbol: str
    decision: str
    confidence: float
    reasoning: Dict[str, Any]
    module_scores: Dict[str, float]
    risk_score: float
    data_quality: float

class DecisionMaker:
    """
    Central decision-making component
    STRICT RULE: Only trade if confidence > 60%
    NO RANDOM TRADES, NO FALLBACKS
    """
    
    def __init__(self, config):
        self.config = config
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        self.decision_history: List[DecisionRecord] = []
        self.logger = logger
        
        # Decision statistics
        self.stats = {
            'total_decisions': 0,
            'trade_decisions': 0,
            'no_trade_decisions': 0,
            'avg_confidence': 0.0,
            'confidence_distribution': []
        }
        
        self.logger.info(f"DecisionMaker initialized with {self.confidence_threshold*100}% threshold")
    
    def make_decision(
        self, 
        symbol: str,
        analysis_results: Dict[str, Any],
        risk_assessment: Dict[str, Any],
        data_quality: float
    ) -> Dict[str, Any]:
        """
        Make trading decision based on all inputs
        
        Returns:
            Dict with keys: decision, confidence, reasoning, signals
        """
        self.stats['total_decisions'] += 1
        
        # Initialize decision result
        result = {
            'decision': 'NO_TRADE',  # Default to NO_TRADE
            'confidence': 0.0,
            'reasoning': {},
            'signals': {},
            'module_breakdown': {}
        }
        
        try:
            # Step 1: Validate inputs
            validation_result = self._validate_inputs(
                analysis_results, risk_assessment, data_quality
            )
            if not validation_result['is_valid']:
                result['reasoning']['validation'] = validation_result['reason']
                return self._finalize_decision(symbol, result, risk_assessment, data_quality)
            
            # Step 2: Check required modules
            missing_modules = self._check_required_modules(analysis_results)
            if missing_modules:
                result['reasoning']['missing_modules'] = f"Missing required: {missing_modules}"
                return self._finalize_decision(symbol, result, risk_assessment, data_quality)
            
            # Step 3: Calculate composite confidence score
            confidence, module_breakdown = self._calculate_confidence(analysis_results)
            result['confidence'] = confidence
            result['module_breakdown'] = module_breakdown
            
            # Step 4: Aggregate signals from all modules
            signals = self._aggregate_signals(analysis_results)
            result['signals'] = signals
            
            # Step 5: Make decision based on confidence threshold
            if confidence >= self.confidence_threshold:
                # Additional validation for TRADE decision
                trade_validation = self._validate_trade_conditions(
                    confidence, signals, risk_assessment
                )
                
                if trade_validation['can_trade']:
                    result['decision'] = 'TRADE'
                    result['reasoning']['decision'] = f"High confidence: {confidence:.2%}"
                    self.stats['trade_decisions'] += 1
                else:
                    result['reasoning']['trade_validation'] = trade_validation['reason']
            else:
                result['reasoning']['confidence'] = f"Below threshold: {confidence:.2%} < {self.confidence_threshold:.2%}"
                self.stats['no_trade_decisions'] += 1
            
            # Step 6: Add market context
            result['market_context'] = self._get_market_context(analysis_results)
            
        except Exception as e:
            self.logger.error(f"Decision making error: {e}")
            result['reasoning']['error'] = f"Decision error: {str(e)}"
        
        return self._finalize_decision(symbol, result, risk_assessment, data_quality)
    
    def _validate_inputs(
        self, 
        analysis_results: Dict[str, Any],
        risk_assessment: Dict[str, Any],
        data_quality: float
    ) -> Dict[str, Any]:
        """Validate all inputs before decision making"""
        
        # Check data quality
        if data_quality < 0.6:
            return {
                'is_valid': False,
                'reason': f'Poor data quality: {data_quality:.2f}'
            }
        
        # Check if we have any analysis results
        if not analysis_results or not isinstance(analysis_results, dict):
            return {
                'is_valid': False,
                'reason': 'No analysis results available'
            }
        
        # Check risk assessment
        if not risk_assessment or risk_assessment.get('risk_score', 1.0) > self.config.trading.max_risk_score:
            return {
                'is_valid': False,
                'reason': f"Risk too high: {risk_assessment.get('risk_score', 1.0):.2f}"
            }
        
        return {'is_valid': True, 'reason': None}
    
    def _check_required_modules(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Check if all required modules have valid results"""
        missing = []
        
        for module in REQUIRED_MODULES:
            if module not in analysis_results:
                missing.append(module)
            elif analysis_results[module].get('status') == 'failed':
                missing.append(f"{module}(failed)")
            elif analysis_results[module].get('confidence', 0) == 0:
                missing.append(f"{module}(no_confidence)")
        
        return missing
    
    def _calculate_confidence(self, analysis_results: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate weighted confidence score from all modules
        NO RANDOM COMPONENTS - Pure calculation based on analysis
        """
        weighted_sum = 0.0
        total_weight = 0.0
        breakdown = {}
        
        for module, weight in MODULE_WEIGHTS.items():
            if module in analysis_results:
                result = analysis_results[module]
                
                # Skip failed modules
                if result.get('status') == 'failed':
                    self.logger.debug(f"Skipping failed module: {module}")
                    continue
                
                # Get module confidence
                module_confidence = result.get('confidence', 0.0)
                
                # Apply data quality factor if available
                if 'data_quality' in result:
                    quality_factor = result['data_quality']
                    module_confidence *= quality_factor
                
                # Calculate weighted contribution
                weighted_sum += module_confidence * weight
                total_weight += weight
                
                # Record breakdown
                breakdown[module] = {
                    'raw_confidence': result.get('confidence', 0.0),
                    'weight': weight,
                    'contribution': module_confidence * weight,
                    'signals': result.get('signals', {})
                }
        
        # Calculate final confidence
        if total_weight > 0:
            confidence = weighted_sum / total_weight
        else:
            confidence = 0.0
        
        # Ensure confidence is in [0, 1] range
        confidence = max(0.0, min(1.0, confidence))
        
        # Update statistics
        self.stats['confidence_distribution'].append(confidence)
        self.stats['avg_confidence'] = np.mean(self.stats['confidence_distribution'][-100:])  # Last 100
        
        self.logger.debug(f"Calculated confidence: {confidence:.2%} from {len(breakdown)} modules")
        
        return confidence, breakdown
    
    def _aggregate_signals(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate all signals from analysis modules"""
        aggregated = {
            'bullish_signals': [],
            'bearish_signals': [],
            'neutral_signals': [],
            'strength': 0.0,
            'direction': None
        }
        
        for module, result in analysis_results.items():
            if result.get('status') == 'failed':
                continue
                
            # Get module signals
            signals = result.get('signals', {})
            direction = result.get('direction')
            
            # Categorize signals
            if direction == 'long' or direction == 'bullish':
                aggregated['bullish_signals'].extend(
                    [f"{module}:{sig}" for sig in signals.keys()]
                )
            elif direction == 'short' or direction == 'bearish':
                aggregated['bearish_signals'].extend(
                    [f"{module}:{sig}" for sig in signals.keys()]
                )
            else:
                aggregated['neutral_signals'].extend(
                    [f"{module}:{sig}" for sig in signals.keys()]
                )
        
        # Calculate overall direction
        bull_count = len(aggregated['bullish_signals'])
        bear_count = len(aggregated['bearish_signals'])
        
        if bull_count > bear_count * 1.5:  # 50% more bullish signals
            aggregated['direction'] = 'long'
            aggregated['strength'] = bull_count / (bull_count + bear_count) if (bull_count + bear_count) > 0 else 0
        elif bear_count > bull_count * 1.5:  # 50% more bearish signals
            aggregated['direction'] = 'short'
            aggregated['strength'] = bear_count / (bull_count + bear_count) if (bull_count + bear_count) > 0 else 0
        else:
            aggregated['direction'] = 'neutral'
            aggregated['strength'] = 0.5
        
        return aggregated
    
    def _validate_trade_conditions(
        self, 
        confidence: float,
        signals: Dict[str, Any],
        risk_assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Additional validation before allowing a trade"""
        
        # Check if we have a clear direction
        if signals.get('direction') == 'neutral':
            return {
                'can_trade': False,
                'reason': 'No clear market direction'
            }
        
        # Check signal strength
        if signals.get('strength', 0) < 0.6:
            return {
                'can_trade': False,
                'reason': f"Weak signal strength: {signals.get('strength', 0):.2%}"
            }
        
        # Check risk parameters
        if risk_assessment.get('max_position_size', 0) == 0:
            return {
                'can_trade': False,
                'reason': 'Risk manager rejected position sizing'
            }
        
        # All conditions met
        return {
            'can_trade': True,
            'reason': 'All trade conditions satisfied'
        }
    
    def _get_market_context(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract market context from analysis results"""
        context = {
            'regime': 'unknown',
            'volatility': 'normal',
            'trend': 'neutral',
            'volume': 'average'
        }
        
        # Get market regime from market analyzer
        if 'market' in analysis_results:
            market_data = analysis_results['market']
            context['regime'] = market_data.get('regime', 'unknown')
            context['volatility'] = market_data.get('volatility_state', 'normal')
            
        # Get trend from technical analyzer
        if 'technical' in analysis_results:
            tech_data = analysis_results['technical']
            context['trend'] = tech_data.get('trend', 'neutral')
            context['volume'] = tech_data.get('volume_state', 'average')
        
        return context
    
    def _finalize_decision(
        self, 
        symbol: str,
        result: Dict[str, Any],
        risk_assessment: Dict[str, Any],
        data_quality: float
    ) -> Dict[str, Any]:
        """Finalize and record the decision"""
        
        # Create decision record
        record = DecisionRecord(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            decision=result['decision'],
            confidence=result['confidence'],
            reasoning=result['reasoning'],
            module_scores=result.get('module_breakdown', {}),
            risk_score=risk_assessment.get('risk_score', 0),
            data_quality=data_quality
        )
        
        # Store in history
        self.decision_history.append(record)
        
        # Log decision
        self.logger.info(
            f"Decision for {symbol}: {result['decision']} "
            f"(confidence: {result['confidence']:.2%})"
        )
        
        # Add decision metadata
        result['timestamp'] = record.timestamp
        result['decision_id'] = f"{symbol}_{record.timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        return result
    
    def get_decision_stats(self) -> Dict[str, Any]:
        """Get decision-making statistics"""
        recent_decisions = self.decision_history[-100:]  # Last 100 decisions
        
        if not recent_decisions:
            return self.stats
        
        # Calculate additional stats
        trade_decisions = [d for d in recent_decisions if d.decision == 'TRADE']
        no_trade_decisions = [d for d in recent_decisions if d.decision == 'NO_TRADE']
        
        stats = {
            **self.stats,
            'recent_trade_rate': len(trade_decisions) / len(recent_decisions) if recent_decisions else 0,
            'avg_trade_confidence': np.mean([d.confidence for d in trade_decisions]) if trade_decisions else 0,
            'avg_no_trade_confidence': np.mean([d.confidence for d in no_trade_decisions]) if no_trade_decisions else 0,
            'confidence_percentiles': {
                'p25': np.percentile([d.confidence for d in recent_decisions], 25),
                'p50': np.percentile([d.confidence for d in recent_decisions], 50),
                'p75': np.percentile([d.confidence for d in recent_decisions], 75),
                'p90': np.percentile([d.confidence for d in recent_decisions], 90)
            }
        }
        
        return stats
    
    def export_decision_history(self, filepath: str):
        """Export decision history for analysis"""
        history_data = []
        
        for record in self.decision_history:
            history_data.append({
                'timestamp': record.timestamp.isoformat(),
                'symbol': record.symbol,
                'decision': record.decision,
                'confidence': record.confidence,
                'risk_score': record.risk_score,
                'data_quality': record.data_quality,
                'reasoning': record.reasoning,
                'module_scores': {
                    module: scores.get('contribution', 0) 
                    for module, scores in record.module_scores.items()
                }
            })
        
        with open(filepath, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        self.logger.info(f"Exported {len(history_data)} decisions to {filepath}")