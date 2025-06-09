"""
BEAST Trading System - Risk Manager
Comprehensive risk management with position sizing and validation
Merged from: risk_monitor.py, risk_validator.py, adaptive_sizing.py
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from collections import deque
import json
from enum import Enum

from config.settings import config
from utils.logger import get_logger

logger = get_logger(__name__)

class RiskLevel(Enum):
    """Risk level classifications"""
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics for a position"""
    symbol: str
    risk_score: float  # 0-1 scale
    risk_level: RiskLevel
    max_position_size: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    correlation_risk: float
    concentration_risk: float
    volatility_risk: float
    liquidity_risk: float
    drawdown_risk: float
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional VaR 95%
    kelly_fraction: float
    warnings: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class AdaptiveSizer:
    """Adaptive position sizing based on market conditions and performance"""
    
    def __init__(self, base_size: float = 0.02):
        self.base_size = base_size
        self.performance_history = deque(maxlen=100)
        self.size_adjustments = deque(maxlen=50)
        self.min_size = 0.001  # 0.1%
        self.max_size = 0.05   # 5%
        
    def calculate_size(
        self,
        confidence: float,
        volatility: float,
        win_rate: float,
        risk_metrics: RiskMetrics
    ) -> float:
        """Calculate adaptive position size"""
        # Start with base size
        size = self.base_size
        
        # Confidence adjustment (0.5 to 1.5x)
        confidence_mult = 0.5 + confidence
        size *= confidence_mult
        
        # Volatility adjustment (inverse relationship)
        vol_mult = 1.0 / (1.0 + volatility * 10)  # Higher vol = smaller size
        size *= vol_mult
        
        # Win rate adjustment
        if len(self.performance_history) >= 20:
            if win_rate > 0.6:
                size *= 1.2
            elif win_rate < 0.4:
                size *= 0.8
        
        # Risk score adjustment
        risk_mult = 1.0 - (risk_metrics.risk_score * 0.5)
        size *= risk_mult
        
        # Kelly Criterion influence (conservative)
        if risk_metrics.kelly_fraction > 0:
            kelly_influence = 0.25  # Use 25% of Kelly suggestion
            kelly_size = risk_metrics.kelly_fraction * kelly_influence
            size = (size * 0.7) + (kelly_size * 0.3)  # Blend approaches
        
        # Apply limits
        size = max(self.min_size, min(size, self.max_size))
        
        # Record adjustment
        self.size_adjustments.append({
            'timestamp': datetime.now(timezone.utc),
            'base': self.base_size,
            'final': size,
            'factors': {
                'confidence': confidence_mult,
                'volatility': vol_mult,
                'win_rate': win_rate,
                'risk_score': risk_mult
            }
        })
        
        return size
    
    def update_performance(self, trade_result: Dict[str, Any]):
        """Update performance history for adaptive sizing"""
        self.performance_history.append({
            'timestamp': trade_result['timestamp'],
            'profit': trade_result['profit'],
            'size_used': trade_result['position_size'],
            'risk_taken': trade_result.get('risk_score', 0.5)
        })

class RiskManager:
    """
    Comprehensive risk management system
    NO BLIND RISKS - Strict validation and limits
    """
    
    def __init__(self, config):
        self.config = config
        self.max_risk_score = config.trading.max_risk_score
        self.max_portfolio_risk = 0.06  # 6% total portfolio risk
        self.max_correlation = config.trading.max_correlation
        
        # Risk tracking
        self.active_risks = {}  # symbol -> RiskMetrics
        self.risk_history = deque(maxlen=1000)
        self.correlation_matrix = pd.DataFrame()
        
        # Adaptive sizing
        self.sizer = AdaptiveSizer(config.trading.max_position_size_pct)
        
        # Performance metrics for risk adjustment
        self.trade_outcomes = deque(maxlen=100)
        self.daily_returns = deque(maxlen=252)  # 1 year of daily returns
        
        # Risk limits and circuit breakers
        self.daily_loss_limit = 0.05  # 5% daily loss limit
        self.consecutive_losses_limit = 5
        self.current_consecutive_losses = 0
        self.daily_pnl = 0.0
        self.last_reset = datetime.now(timezone.utc).date()
        
        logger.info("RiskManager initialized with adaptive sizing")
    
    async def assess(
        self,
        symbol: str,
        analysis: Dict[str, Any],
        portfolio: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Comprehensive risk assessment
        Returns risk metrics and position sizing
        """
        try:
            # Reset daily counters if needed
            self._check_daily_reset()
            
            # Check circuit breakers first
            if self._circuit_breaker_triggered():
                return self._create_reject_response("Circuit breaker active")
            
            # Calculate individual risk components
            market_risk = self._assess_market_risk(analysis)
            volatility_risk = self._assess_volatility_risk(analysis)
            liquidity_risk = self._assess_liquidity_risk(symbol, analysis)
            correlation_risk = self._assess_correlation_risk(symbol, portfolio)
            concentration_risk = self._assess_concentration_risk(symbol, portfolio)
            drawdown_risk = self._assess_drawdown_risk(portfolio)
            
            # Calculate composite risk score
            risk_score = self._calculate_composite_risk(
                market_risk,
                volatility_risk,
                liquidity_risk,
                correlation_risk,
                concentration_risk,
                drawdown_risk
            )
            
            # Determine risk level
            risk_level = self._classify_risk_level(risk_score)
            
            # Calculate VaR and CVaR
            var_95, cvar_95 = self._calculate_var_metrics(analysis, portfolio)
            
            # Calculate Kelly fraction
            kelly_fraction = self._calculate_kelly_fraction(analysis)
            
            # Create risk metrics
            risk_metrics = RiskMetrics(
                symbol=symbol,
                risk_score=risk_score,
                risk_level=risk_level,
                max_position_size=0.0,  # Set later
                stop_loss=0.0,  # Set later
                take_profit=0.0,  # Set later
                risk_reward_ratio=0.0,  # Set later
                correlation_risk=correlation_risk,
                concentration_risk=concentration_risk,
                volatility_risk=volatility_risk,
                liquidity_risk=liquidity_risk,
                drawdown_risk=drawdown_risk,
                var_95=var_95,
                cvar_95=cvar_95,
                kelly_fraction=kelly_fraction
            )
            
            # Check if risk is acceptable
            if risk_score > self.max_risk_score:
                risk_metrics.warnings.append(f"Risk score {risk_score:.2f} exceeds limit {self.max_risk_score}")
                return self._create_reject_response("Risk score too high", risk_metrics)
            
            # Calculate position parameters
            position_params = self._calculate_position_parameters(
                symbol,
                analysis,
                risk_metrics,
                portfolio
            )
            
            # Update risk metrics with position parameters
            risk_metrics.max_position_size = position_params['position_size']
            risk_metrics.stop_loss = position_params['stop_loss']
            risk_metrics.take_profit = position_params['take_profit']
            risk_metrics.risk_reward_ratio = position_params['risk_reward_ratio']
            
            # Validate total portfolio risk
            portfolio_risk_check = self._validate_portfolio_risk(
                risk_metrics,
                portfolio
            )
            
            if not portfolio_risk_check['acceptable']:
                return self._create_reject_response(
                    portfolio_risk_check['reason'],
                    risk_metrics
                )
            
            # Store active risk
            self.active_risks[symbol] = risk_metrics
            
            # Record risk assessment
            self._record_risk_assessment(risk_metrics)
            
            # Create response
            return {
                'risk_score': risk_score,
                'risk_level': risk_level.value,
                'acceptable': True,
                'max_position_size': risk_metrics.max_position_size,
                'stop_loss': risk_metrics.stop_loss,
                'take_profit': risk_metrics.take_profit,
                'risk_reward_ratio': risk_metrics.risk_reward_ratio,
                'risk_metrics': risk_metrics,
                'warnings': risk_metrics.warnings
            }
            
        except Exception as e:
            logger.error(f"Risk assessment error: {e}")
            return self._create_reject_response(f"Assessment error: {str(e)}")
    
    def _assess_market_risk(self, analysis: Dict[str, Any]) -> float:
        """Assess market-related risks"""
        risk = 0.0
        
        # Check for conflicting signals
        signals = []
        for module in ['technical', 'patterns', 'blockchain', 'market']:
            if module in analysis:
                direction = analysis[module].get('direction')
                if direction:
                    signals.append(direction)
        
        # Conflicting signals increase risk
        if signals and len(set(signals)) > 1:
            risk += 0.3
        
        # Low confidence across modules
        confidences = []
        for module, result in analysis.items():
            if isinstance(result, dict) and 'confidence' in result:
                confidences.append(result['confidence'])
        
        if confidences:
            avg_confidence = np.mean(confidences)
            if avg_confidence < 0.5:
                risk += 0.2
        
        # Market regime risk
        if 'market' in analysis:
            regime = analysis['market'].get('regime', 'unknown')
            if regime == 'high_volatility':
                risk += 0.2
            elif regime == 'bear_trend':
                risk += 0.15
        
        return min(risk, 1.0)
    
    def _assess_volatility_risk(self, analysis: Dict[str, Any]) -> float:
        """Assess volatility-based risk"""
        risk = 0.0
        
        # Get volatility metrics
        volatility = 0.02  # Default 2%
        
        if 'technical' in analysis:
            tech = analysis['technical']
            if 'volatility' in tech.get('indicators', {}):
                volatility = tech['indicators']['volatility']
            elif 'ATR_14' in tech.get('indicators', {}):
                atr = tech['indicators']['ATR_14']
                price = tech['indicators'].get('close', 1)
                volatility = atr / price if price > 0 else 0.02
        
        # Volatility risk scaling
        if volatility > 0.05:  # >5% volatility
            risk = 0.8
        elif volatility > 0.03:  # >3% volatility
            risk = 0.5
        elif volatility > 0.02:  # >2% volatility
            risk = 0.3
        else:
            risk = 0.1
        
        return risk
    
    def _assess_liquidity_risk(self, symbol: str, analysis: Dict[str, Any]) -> float:
        """Assess liquidity risk"""
        risk = 0.0
        
        # Check volume
        if 'technical' in analysis:
            volume_ratio = analysis['technical'].get('indicators', {}).get('volume_ratio', 1.0)
            
            if volume_ratio < 0.5:  # Low volume
                risk += 0.3
            
        # Check spread (if available)
        if 'market' in analysis:
            spread_pct = analysis['market'].get('spread_pct', 0)
            if spread_pct > 0.5:  # >0.5% spread
                risk += 0.2
        
        # Check if it's a major pair
        major_pairs = ['BTC', 'ETH', 'BNB', 'SOL', 'ADA']
        if not any(pair in symbol for pair in major_pairs):
            risk += 0.1
        
        return min(risk, 1.0)
    
    def _assess_correlation_risk(self, symbol: str, portfolio: Dict[str, Any]) -> float:
        """Assess correlation risk with existing positions"""
        if not portfolio:
            return 0.0
        
        # Simplified correlation assessment
        # In production, calculate actual correlation matrix
        similar_positions = 0
        
        for position_symbol in portfolio.keys():
            # Check if same base asset
            if position_symbol.split('-')[0] == symbol.split('-')[0]:
                similar_positions += 1
        
        # Risk increases with correlated positions
        if similar_positions >= 3:
            return 0.7
        elif similar_positions >= 2:
            return 0.5
        elif similar_positions >= 1:
            return 0.3
        
        return 0.0
    
    def _assess_concentration_risk(self, symbol: str, portfolio: Dict[str, Any]) -> float:
        """Assess portfolio concentration risk"""
        if not portfolio:
            return 0.0
        
        # Calculate current portfolio allocation
        total_value = sum(pos.get('value', 0) for pos in portfolio.values())
        
        if total_value == 0:
            return 0.0
        
        # Check single position concentration
        largest_position = max(portfolio.values(), key=lambda x: x.get('value', 0), default={})
        if largest_position:
            concentration = largest_position.get('value', 0) / total_value
            
            if concentration > 0.3:  # >30% in one position
                return 0.8
            elif concentration > 0.2:  # >20%
                return 0.5
            elif concentration > 0.15:  # >15%
                return 0.3
        
        return 0.1
    
    def _assess_drawdown_risk(self, portfolio: Dict[str, Any]) -> float:
        """Assess drawdown risk"""
        if not self.daily_returns:
            return 0.2  # Default moderate risk
        
        # Calculate max drawdown from daily returns
        returns = list(self.daily_returns)
        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(np.min(drawdown))
        
        # Risk based on drawdown
        if max_drawdown > 0.2:  # >20% drawdown
            return 0.9
        elif max_drawdown > 0.15:  # >15%
            return 0.7
        elif max_drawdown > 0.1:  # >10%
            return 0.5
        elif max_drawdown > 0.05:  # >5%
            return 0.3
        
        return 0.1
    
    def _calculate_composite_risk(self, *risk_components: float) -> float:
        """Calculate weighted composite risk score"""
        weights = [0.25, 0.20, 0.15, 0.15, 0.15, 0.10]  # Adjust weights as needed
        
        weighted_sum = sum(w * r for w, r in zip(weights, risk_components))
        
        # Non-linear scaling for extreme risks
        if any(r > 0.8 for r in risk_components):
            weighted_sum *= 1.2
        
        return min(weighted_sum, 1.0)
    
    def _classify_risk_level(self, risk_score: float) -> RiskLevel:
        """Classify risk score into risk level"""
        if risk_score < 0.2:
            return RiskLevel.MINIMAL
        elif risk_score < 0.4:
            return RiskLevel.LOW
        elif risk_score < 0.6:
            return RiskLevel.MODERATE
        elif risk_score < 0.8:
            return RiskLevel.HIGH
        else:
            return RiskLevel.EXTREME
    
    def _calculate_var_metrics(
        self,
        analysis: Dict[str, Any],
        portfolio: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Calculate Value at Risk and Conditional VaR"""
        # Simplified VaR calculation
        # In production, use historical simulation or Monte Carlo
        
        volatility = 0.02  # Default
        if 'technical' in analysis:
            volatility = analysis['technical'].get('indicators', {}).get('volatility', 0.02)
        
        # Parametric VaR (95% confidence)
        z_score = 1.645  # 95% confidence
        var_95 = volatility * z_score
        
        # CVaR approximation (expected loss beyond VaR)
        cvar_95 = var_95 * 1.4  # Rough approximation
        
        return var_95, cvar_95
    
    def _calculate_kelly_fraction(self, analysis: Dict[str, Any]) -> float:
        """Calculate Kelly Criterion fraction for position sizing"""
        if len(self.trade_outcomes) < 20:
            return 0.0  # Not enough data
        
        # Calculate win rate and average win/loss
        wins = [t for t in self.trade_outcomes if t['profit'] > 0]
        losses = [t for t in self.trade_outcomes if t['profit'] <= 0]
        
        if not wins or not losses:
            return 0.0
        
        win_rate = len(wins) / len(self.trade_outcomes)
        avg_win = np.mean([t['profit'] for t in wins])
        avg_loss = abs(np.mean([t['profit'] for t in losses]))
        
        if avg_loss == 0:
            return 0.0
        
        # Kelly formula: f = p - q/b
        # p = win probability, q = loss probability, b = win/loss ratio
        b = avg_win / avg_loss
        q = 1 - win_rate
        
        kelly = win_rate - (q / b)
        
        # Conservative Kelly (25% of full Kelly)
        return max(0, min(kelly * 0.25, 0.25))
    
    def _calculate_position_parameters(
        self,
        symbol: str,
        analysis: Dict[str, Any],
        risk_metrics: RiskMetrics,
        portfolio: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate position size and risk parameters"""
        # Get confidence
        confidence = 0.5  # Default
        if 'technical' in analysis:
            confidence = analysis['technical'].get('confidence', 0.5)
        
        # Get volatility
        volatility = risk_metrics.volatility_risk
        
        # Calculate win rate
        win_rate = 0.5  # Default
        if self.trade_outcomes:
            wins = sum(1 for t in self.trade_outcomes if t['profit'] > 0)
            win_rate = wins / len(self.trade_outcomes)
        
        # Calculate adaptive position size
        position_size = self.sizer.calculate_size(
            confidence,
            volatility,
            win_rate,
            risk_metrics
        )
        
        # Calculate stop loss based on volatility
        atr_multiplier = 2.0  # 2x ATR for stop loss
        stop_loss = volatility * atr_multiplier
        
        # Adjust stop loss based on risk level
        if risk_metrics.risk_level == RiskLevel.HIGH:
            stop_loss *= 0.7  # Tighter stop
        elif risk_metrics.risk_level == RiskLevel.EXTREME:
            stop_loss *= 0.5  # Very tight stop
        
        # Minimum stop loss
        stop_loss = max(stop_loss, self.config.trading.stop_loss_pct)
        
        # Calculate take profit for desired risk/reward
        min_rr_ratio = 2.0  # Minimum 2:1 risk/reward
        take_profit = stop_loss * min_rr_ratio
        
        # Adjust based on market conditions
        if 'market' in analysis:
            regime = analysis['market'].get('regime', 'unknown')
            if regime == 'trending':
                take_profit *= 1.5  # Let profits run in trends
            elif regime == 'ranging':
                take_profit *= 0.8  # Take profits quicker in ranges
        
        return {
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward_ratio': take_profit / stop_loss
        }
    
    def _validate_portfolio_risk(
        self,
        risk_metrics: RiskMetrics,
        portfolio: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate total portfolio risk"""
        # Calculate current portfolio risk
        total_risk = 0.0
        
        for symbol, position in portfolio.items():
            if symbol in self.active_risks:
                position_risk = (
                    position.get('size', 0) * 
                    self.active_risks[symbol].stop_loss
                )
                total_risk += position_risk
        
        # Add new position risk
        new_position_risk = risk_metrics.max_position_size * risk_metrics.stop_loss
        total_risk += new_position_risk
        
        # Check against limit
        if total_risk > self.max_portfolio_risk:
            return {
                'acceptable': False,
                'reason': f'Total portfolio risk {total_risk:.2%} exceeds limit {self.max_portfolio_risk:.2%}'
            }
        
        # Check position count
        if len(portfolio) >= 10:
            return {
                'acceptable': False,
                'reason': 'Maximum position count reached'
            }
        
        return {'acceptable': True, 'reason': None}
    
    def _circuit_breaker_triggered(self) -> bool:
        """Check if any circuit breakers are active"""
        # Daily loss limit
        if self.daily_pnl <= -self.daily_loss_limit:
            logger.warning(f"Daily loss limit reached: {self.daily_pnl:.2%}")
            return True
        
        # Consecutive losses
        if self.current_consecutive_losses >= self.consecutive_losses_limit:
            logger.warning(f"Consecutive loss limit reached: {self.current_consecutive_losses}")
            return True
        
        return False
    
    def _check_daily_reset(self):
        """Reset daily counters if new day"""
        current_date = datetime.now(timezone.utc).date()
        
        if current_date > self.last_reset:
            self.daily_pnl = 0.0
            self.last_reset = current_date
            logger.info("Daily risk counters reset")
    
    def _create_reject_response(
        self,
        reason: str,
        risk_metrics: Optional[RiskMetrics] = None
    ) -> Dict[str, Any]:
        """Create a risk rejection response"""
        return {
            'risk_score': risk_metrics.risk_score if risk_metrics else 1.0,
            'risk_level': risk_metrics.risk_level.value if risk_metrics else 'extreme',
            'acceptable': False,
            'max_position_size': 0.0,
            'stop_loss': 0.0,
            'take_profit': 0.0,
            'risk_reward_ratio': 0.0,
            'rejection_reason': reason,
            'risk_metrics': risk_metrics
        }
    
    def _record_risk_assessment(self, risk_metrics: RiskMetrics):
        """Record risk assessment for analysis"""
        self.risk_history.append({
            'timestamp': risk_metrics.timestamp,
            'symbol': risk_metrics.symbol,
            'risk_score': risk_metrics.risk_score,
            'risk_level': risk_metrics.risk_level.value,
            'position_size': risk_metrics.max_position_size,
            'warnings': len(risk_metrics.warnings)
        })
    
    def update_trade_outcome(self, trade_result: Dict[str, Any]):
        """Update trade outcome for risk adjustment"""
        self.trade_outcomes.append(trade_result)
        
        # Update consecutive losses
        if trade_result['profit'] <= 0:
            self.current_consecutive_losses += 1
        else:
            self.current_consecutive_losses = 0
        
        # Update daily PnL
        self.daily_pnl += trade_result['return_pct']
        
        # Update daily returns
        self.daily_returns.append(trade_result['return_pct'])
        
        # Update adaptive sizer
        self.sizer.update_performance(trade_result)
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get current risk summary"""
        return {
            'active_positions': len(self.active_risks),
            'total_risk_score': np.mean([r.risk_score for r in self.active_risks.values()]) if self.active_risks else 0,
            'highest_risk': max(self.active_risks.values(), key=lambda x: x.risk_score).symbol if self.active_risks else None,
            'daily_pnl': self.daily_pnl,
            'consecutive_losses': self.current_consecutive_losses,
            'circuit_breakers_active': self._circuit_breaker_triggered(),
            'current_limits': {
                'daily_loss_limit': self.daily_loss_limit,
                'max_positions': 10,
                'max_risk_score': self.max_risk_score
            }
        }
    
    def export_risk_report(self, filepath: str):
        """Export detailed risk report"""
        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'summary': self.get_risk_summary(),
            'active_risks': {
                symbol: {
                    'risk_score': metrics.risk_score,
                    'risk_level': metrics.risk_level.value,
                    'position_size': metrics.max_position_size,
                    'stop_loss': metrics.stop_loss,
                    'var_95': metrics.var_95
                }
                for symbol, metrics in self.active_risks.items()
            },
            'performance': {
                'total_trades': len(self.trade_outcomes),
                'win_rate': sum(1 for t in self.trade_outcomes if t['profit'] > 0) / len(self.trade_outcomes) if self.trade_outcomes else 0,
                'avg_risk_taken': np.mean([t.get('risk_score', 0.5) for t in self.trade_outcomes]) if self.trade_outcomes else 0
            },
            'size_adjustments': list(self.sizer.size_adjustments)[-10:]  # Last 10
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Risk report exported to {filepath}")
    
    def is_healthy(self) -> bool:
        """Check if risk manager is healthy"""
        return not self._circuit_breaker_triggered()