"""
BEAST Trading System - Production Strategy Module
Live trading strategy selection and management with real-time adaptation
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
import numpy as np
import json
import asyncio
from enum import Enum
from collections import defaultdict, deque
import threading
import pickle

from config.settings import config
from utils.logger import get_logger
from utils.metrics import MetricsCollector
import strategy_rules

logger = get_logger(__name__)

class StrategyType(Enum):
    """Strategy type enumeration"""
    SPOT = "spot"
    FUTURES = "futures"
    OPTIONS = "options"
    ARBITRAGE = "arbitrage"
    MARKET_MAKING = "market_making"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY = "volatility"
    EVENT_DRIVEN = "event_driven"

class ExecutionMode(Enum):
    """Execution mode for strategies"""
    AGGRESSIVE = "aggressive"
    NORMAL = "normal"
    CONSERVATIVE = "conservative"
    PASSIVE = "passive"

@dataclass
class LiveStrategyState:
    """Live strategy state tracking"""
    strategy_name: str
    entry_time: datetime
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    position_size: float
    stop_loss: float
    take_profit: float
    trailing_stop: Optional[float] = None
    highest_price: float = 0.0
    lowest_price: float = float('inf')
    adjustments: List[Dict[str, Any]] = field(default_factory=list)
    signals: Set[str] = field(default_factory=set)
    
    def update_price(self, price: float):
        """Update current price and track extremes"""
        self.current_price = price
        self.highest_price = max(self.highest_price, price)
        self.lowest_price = min(self.lowest_price, price)
        self.unrealized_pnl = (price - self.entry_price) * self.position_size

@dataclass
class StrategyPerformance:
    """Real-time strategy performance metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    daily_returns: deque = field(default_factory=lambda: deque(maxlen=30))

class ProductionStrategyManager:
    """
    Production-ready strategy management for live trading
    Handles strategy selection, execution, monitoring, and adaptation
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logger
        
        # Strategy definitions with production parameters
        self.strategies = self._initialize_strategies()
        
        # Live tracking
        self.active_strategies: Dict[str, LiveStrategyState] = {}
        self.strategy_performance: Dict[str, StrategyPerformance] = {}
        self.execution_queue: asyncio.Queue = asyncio.Queue()
        
        # Risk limits
        self.max_concurrent_strategies = 5
        self.max_strategy_allocation = 0.2  # 20% max per strategy
        self.daily_loss_limit = 0.05  # 5% daily loss limit
        self.strategy_timeout = 86400  # 24 hours default
        
        # Performance tracking
        self.metrics = MetricsCollector()
        self.performance_window = deque(maxlen=1000)
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Load historical performance
        self._load_performance_data()
        
        # Start monitoring
        self.monitoring_task = None
        
        logger.info(f"Production Strategy Manager initialized with {len(self.strategies)} strategies")
    
    def _initialize_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize production strategy configurations"""
        strategies = {}
        
        # Crypto-specific strategies
        if self.config.trading.strategy_config['enable_crypto_strategies']:
            strategies.update({
                'funding_arbitrage': {
                    'logic': strategy_rules.funding_rate_arbitrage_logic,
                    'type': StrategyType.ARBITRAGE,
                    'execution_mode': ExecutionMode.AGGRESSIVE,
                    'min_confidence': 0.65,
                    'max_allocation': 0.15,
                    'timeout': 3600,  # 1 hour
                    'risk_params': {
                        'stop_loss': 0.005,  # 0.5%
                        'take_profit': 0.015,  # 1.5%
                        'max_slippage': 0.001
                    }
                },
                'perp_momentum': {
                    'logic': strategy_rules.perpetual_futures_momentum_logic,
                    'type': StrategyType.MOMENTUM,
                    'execution_mode': ExecutionMode.NORMAL,
                    'min_confidence': 0.70,
                    'max_allocation': 0.10,
                    'timeout': 14400,  # 4 hours
                    'risk_params': {
                        'stop_loss': 0.02,
                        'take_profit': 0.06,
                        'trailing_stop': 0.015,
                        'max_slippage': 0.002
                    }
                },
                'cross_exchange_arb': {
                    'logic': strategy_rules.cross_exchange_arbitrage_logic,
                    'type': StrategyType.ARBITRAGE,
                    'execution_mode': ExecutionMode.AGGRESSIVE,
                    'min_confidence': 0.80,
                    'max_allocation': 0.20,
                    'timeout': 300,  # 5 minutes
                    'risk_params': {
                        'stop_loss': 0.002,
                        'take_profit': 0.005,
                        'max_slippage': 0.0005
                    }
                },
                'adaptive_momentum': {
                    'logic': strategy_rules.adaptive_momentum_logic,
                    'type': StrategyType.MOMENTUM,
                    'execution_mode': ExecutionMode.NORMAL,
                    'min_confidence': 0.68,
                    'max_allocation': 0.12,
                    'timeout': 28800,  # 8 hours
                    'risk_params': {
                        'stop_loss': 0.025,
                        'take_profit': 0.075,
                        'trailing_stop': 0.02,
                        'scale_in': True,
                        'max_slippage': 0.003
                    }
                }
            })
        
        # Market making strategies
        strategies.update({
            'market_making': {
                'logic': strategy_rules.market_making_hft_logic,
                'type': StrategyType.MARKET_MAKING,
                'execution_mode': ExecutionMode.PASSIVE,
                'min_confidence': 0.60,
                'max_allocation': 0.05,
                'timeout': 3600,
                'risk_params': {
                    'spread': 0.001,
                    'inventory_limit': 0.02,
                    'max_order_size': 0.01,
                    'rebalance_threshold': 0.7
                }
            }
        })
        
        # Event-driven strategies
        strategies.update({
            'event_sniper': {
                'logic': strategy_rules.event_sniper_logic,
                'type': StrategyType.EVENT_DRIVEN,
                'execution_mode': ExecutionMode.AGGRESSIVE,
                'min_confidence': 0.75,
                'max_allocation': 0.08,
                'timeout': 1800,  # 30 minutes
                'risk_params': {
                    'stop_loss': 0.03,
                    'take_profit': 0.10,
                    'time_stop': 900  # 15 minutes
                }
            }
        })
        
        # Statistical arbitrage
        strategies.update({
            'stat_arb': {
                'logic': strategy_rules.statistical_arb_logic,
                'type': StrategyType.MEAN_REVERSION,
                'execution_mode': ExecutionMode.NORMAL,
                'min_confidence': 0.72,
                'max_allocation': 0.15,
                'timeout': 7200,  # 2 hours
                'risk_params': {
                    'stop_loss': 0.015,
                    'take_profit': 0.03,
                    'mean_reversion_threshold': 2.0,  # 2 std devs
                    'max_holding_period': 3600
                }
            }
        })
        
        return strategies
    
    async def select_strategy(
        self,
        analysis_bundle: Dict[str, Any],
        risk_profile: Dict[str, Any],
        capital_available: float
    ) -> Optional[Dict[str, Any]]:
        """
        Select optimal strategy for current market conditions
        Production-grade selection with all safety checks
        """
        try:
            # Check if we can take new positions
            if not await self._can_open_position(capital_available):
                logger.info("Cannot open new position: limits reached")
                return None
            
            # Evaluate all strategies
            viable_strategies = await self._evaluate_strategies(
                analysis_bundle,
                risk_profile
            )
            
            if not viable_strategies:
                logger.debug("No viable strategies found")
                return None
            
            # Select best strategy
            selected = await self._select_optimal_strategy(
                viable_strategies,
                risk_profile,
                capital_available
            )
            
            if not selected:
                return None
            
            # Prepare execution parameters
            execution_params = await self._prepare_execution(
                selected,
                analysis_bundle,
                risk_profile,
                capital_available
            )
            
            # Validate execution params
            if not self._validate_execution_params(execution_params):
                logger.error("Invalid execution parameters")
                return None
            
            logger.info(
                f"Selected strategy: {execution_params['strategy_name']} "
                f"confidence: {execution_params['confidence']:.2%} "
                f"allocation: ${execution_params['position_size']:.2f}"
            )
            
            return execution_params
            
        except Exception as e:
            logger.error(f"Strategy selection error: {e}")
            return None
    
    async def _can_open_position(self, capital_available: float) -> bool:
        """Check if we can open a new position"""
        with self.lock:
            # Check concurrent strategies limit
            if len(self.active_strategies) >= self.max_concurrent_strategies:
                return False
            
            # Check daily loss limit
            daily_pnl = self._calculate_daily_pnl()
            if daily_pnl < -self.daily_loss_limit * capital_available:
                logger.warning(f"Daily loss limit reached: {daily_pnl:.2f}")
                return False
            
            # Check available capital
            allocated = sum(s.position_size for s in self.active_strategies.values())
            if allocated >= capital_available * 0.9:  # 90% allocated
                return False
            
            return True
    
    async def _evaluate_strategies(
        self,
        analysis_bundle: Dict[str, Any],
        risk_profile: Dict[str, Any]
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """Evaluate all strategies for current conditions"""
        viable = []
        
        # Add live market data to bundle
        analysis_bundle['live_data'] = {
            'timestamp': datetime.now(timezone.utc),
            'active_positions': len(self.active_strategies),
            'market_hours': self._is_market_hours(analysis_bundle.get('symbol', ''))
        }
        
        for name, strategy in self.strategies.items():
            try:
                # Skip if strategy is currently active
                if name in self.active_strategies:
                    continue
                
                # Check if strategy logic triggers
                if strategy['logic'](analysis_bundle):
                    # Calculate live confidence
                    confidence = await self._calculate_live_confidence(
                        name,
                        strategy,
                        analysis_bundle,
                        risk_profile
                    )
                    
                    if confidence >= strategy['min_confidence']:
                        viable.append((name, strategy, confidence))
                        
            except Exception as e:
                logger.error(f"Error evaluating {name}: {e}")
                continue
        
        return viable
    
    async def _calculate_live_confidence(
        self,
        strategy_name: str,
        strategy_config: Dict[str, Any],
        analysis_bundle: Dict[str, Any],
        risk_profile: Dict[str, Any]
    ) -> float:
        """Calculate real-time confidence with live adjustments"""
        base_confidence = 0.5
        
        # Technical signal strength
        tech_confidence = analysis_bundle.get('technical', {}).get('confidence', 0)
        base_confidence += tech_confidence * 0.2
        
        # Pattern confirmation
        pattern_confidence = analysis_bundle.get('patterns', {}).get('confidence', 0)
        base_confidence += pattern_confidence * 0.15
        
        # Market regime alignment
        regime = analysis_bundle.get('market', {}).get('regime', 'unknown')
        if self._is_regime_aligned(strategy_config['type'], regime):
            base_confidence += 0.1
        
        # Historical performance adjustment
        if strategy_name in self.strategy_performance:
            perf = self.strategy_performance[strategy_name]
            if perf.total_trades >= 20:
                # Adjust based on recent performance
                if perf.win_rate > 0.6:
                    base_confidence += 0.1
                elif perf.win_rate < 0.4:
                    base_confidence -= 0.1
                
                # Sharpe ratio adjustment
                if perf.sharpe_ratio > 1.5:
                    base_confidence += 0.05
        
        # Risk adjustment
        risk_score = risk_profile.get('risk_score', 0.5)
        if risk_score > 0.7:
            base_confidence *= 0.9
        
        # Time of day adjustment (for certain strategies)
        if strategy_config['type'] == StrategyType.MOMENTUM:
            hour = datetime.now(timezone.utc).hour
            if 13 <= hour <= 20:  # Peak trading hours
                base_confidence += 0.05
        
        # Volatility adjustment
        current_volatility = analysis_bundle.get('market', {}).get('volatility', 0.02)
        if strategy_config['type'] == StrategyType.VOLATILITY:
            if current_volatility > 0.03:
                base_confidence += 0.1
        elif current_volatility > 0.05:  # Extreme volatility
            base_confidence *= 0.8
        
        return min(1.0, max(0.0, base_confidence))
    
    def _is_regime_aligned(self, strategy_type: StrategyType, regime: str) -> bool:
        """Check if strategy type aligns with market regime"""
        alignments = {
            StrategyType.MOMENTUM: ['trending', 'breakout'],
            StrategyType.MEAN_REVERSION: ['ranging', 'mean_reverting'],
            StrategyType.VOLATILITY: ['volatile', 'uncertain'],
            StrategyType.ARBITRAGE: ['any'],  # Works in any regime
            StrategyType.MARKET_MAKING: ['ranging', 'low_volatility']
        }
        
        return regime in alignments.get(strategy_type, [])
    
    async def _select_optimal_strategy(
        self,
        viable_strategies: List[Tuple[str, Dict[str, Any], float]],
        risk_profile: Dict[str, Any],
        capital_available: float
    ) -> Optional[Tuple[str, Dict[str, Any], float]]:
        """Select optimal strategy from viable options"""
        if not viable_strategies:
            return None
        
        # Score each strategy
        scored_strategies = []
        
        for name, strategy, confidence in viable_strategies:
            score = confidence
            
            # Risk-adjusted scoring
            risk_score = risk_profile.get('risk_score', 0.5)
            if strategy['type'] in [StrategyType.ARBITRAGE, StrategyType.MARKET_MAKING]:
                score *= 1.1  # Prefer lower risk strategies
            
            # Performance-based scoring
            if name in self.strategy_performance:
                perf = self.strategy_performance[name]
                if perf.total_trades >= 10:
                    score *= (1 + perf.sharpe_ratio * 0.1)
            
            # Capital efficiency scoring
            max_position = capital_available * strategy['max_allocation']
            if max_position < 1000:  # Too small for strategy
                score *= 0.5
            
            scored_strategies.append((score, name, strategy, confidence))
        
        # Sort by score and return best
        scored_strategies.sort(reverse=True)
        
        if scored_strategies:
            _, name, strategy, confidence = scored_strategies[0]
            return (name, strategy, confidence)
        
        return None
    
    async def _prepare_execution(
        self,
        selected: Tuple[str, Dict[str, Any], float],
        analysis_bundle: Dict[str, Any],
        risk_profile: Dict[str, Any],
        capital_available: float
    ) -> Dict[str, Any]:
        """Prepare detailed execution parameters"""
        name, strategy, confidence = selected
        
        # Calculate position size
        position_size = self._calculate_position_size(
            strategy,
            confidence,
            risk_profile,
            capital_available
        )
        
        # Get current price
        current_price = analysis_bundle.get('market_data', {}).get('last_price', 0)
        
        # Calculate risk parameters
        risk_params = self._calculate_risk_parameters(
            strategy,
            current_price,
            analysis_bundle
        )
        
        # Determine execution mode
        exec_mode = self._determine_execution_mode(
            strategy,
            analysis_bundle,
            confidence
        )
        
        return {
            'strategy_name': name,
            'strategy_type': strategy['type'].value,
            'confidence': confidence,
            'position_size': position_size,
            'entry_price': current_price,
            'execution_mode': exec_mode.value,
            'risk_params': risk_params,
            'timeout': strategy.get('timeout', self.strategy_timeout),
            'signals': self._extract_signals(analysis_bundle),
            'metadata': {
                'selected_at': datetime.now(timezone.utc),
                'market_conditions': self._get_market_conditions(analysis_bundle),
                'expected_duration': strategy.get('timeout', 3600)
            }
        }
    
    def _calculate_position_size(
        self,
        strategy: Dict[str, Any],
        confidence: float,
        risk_profile: Dict[str, Any],
        capital_available: float
    ) -> float:
        """Calculate optimal position size with safety checks"""
        # Base allocation
        max_allocation = min(
            strategy['max_allocation'],
            self.max_strategy_allocation
        )
        
        base_size = capital_available * max_allocation
        
        # Confidence adjustment
        confidence_multiplier = min(1.0, confidence / 0.7)
        
        # Risk adjustment
        risk_multiplier = 1.0 - (risk_profile.get('risk_score', 0.5) * 0.3)
        
        # Kelly Criterion (simplified)
        if strategy['type'] in [StrategyType.MOMENTUM, StrategyType.MEAN_REVERSION]:
            win_rate = self._get_strategy_win_rate(strategy['type'])
            if win_rate > 0:
                kelly_fraction = (win_rate - (1 - win_rate)) / 1
                kelly_multiplier = min(0.25, kelly_fraction)  # Cap at 25%
                risk_multiplier *= kelly_multiplier
        
        # Calculate final size
        position_size = base_size * confidence_multiplier * risk_multiplier
        
        # Apply limits
        min_size = 100  # $100 minimum
        max_size = capital_available * 0.2  # 20% max single position
        
        return max(min_size, min(position_size, max_size))
    
    def _calculate_risk_parameters(
        self,
        strategy: Dict[str, Any],
        current_price: float,
        analysis_bundle: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate dynamic risk parameters"""
        base_params = strategy['risk_params'].copy()
        
        # Adjust for volatility
        volatility = analysis_bundle.get('market', {}).get('volatility', 0.02)
        vol_multiplier = max(1.0, volatility / 0.02)
        
        # Dynamic stop loss
        if 'stop_loss' in base_params:
            base_params['stop_loss'] *= vol_multiplier
            base_params['stop_loss_price'] = current_price * (1 - base_params['stop_loss'])
        
        # Dynamic take profit
        if 'take_profit' in base_params:
            base_params['take_profit'] *= vol_multiplier
            base_params['take_profit_price'] = current_price * (1 + base_params['take_profit'])
        
        # Trailing stop
        if 'trailing_stop' in base_params:
            base_params['trailing_stop'] *= vol_multiplier
            base_params['trailing_activated'] = False
        
        # Time-based stops
        if strategy['type'] == StrategyType.EVENT_DRIVEN:
            base_params['time_stop'] = min(
                base_params.get('time_stop', 3600),
                strategy.get('timeout', 3600) * 0.5
            )
        
        return base_params
    
    def _determine_execution_mode(
        self,
        strategy: Dict[str, Any],
        analysis_bundle: Dict[str, Any],
        confidence: float
    ) -> ExecutionMode:
        """Determine optimal execution mode"""
        base_mode = strategy.get('execution_mode', ExecutionMode.NORMAL)
        
        # High confidence -> more aggressive
        if confidence > 0.85:
            if base_mode == ExecutionMode.NORMAL:
                return ExecutionMode.AGGRESSIVE
        
        # High volatility -> more conservative
        volatility = analysis_bundle.get('market', {}).get('volatility', 0.02)
        if volatility > 0.04:
            if base_mode == ExecutionMode.AGGRESSIVE:
                return ExecutionMode.NORMAL
            elif base_mode == ExecutionMode.NORMAL:
                return ExecutionMode.CONSERVATIVE
        
        # Low liquidity -> passive execution
        liquidity = analysis_bundle.get('orderbook', {}).get('liquidity_score', 1.0)
        if liquidity < 0.5:
            return ExecutionMode.PASSIVE
        
        return base_mode
    
    def _validate_execution_params(self, params: Dict[str, Any]) -> bool:
        """Validate execution parameters before trading"""
        required_fields = [
            'strategy_name', 'position_size', 'entry_price',
            'risk_params', 'confidence'
        ]
        
        # Check required fields
        for field in required_fields:
            if field not in params or params[field] is None:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Validate position size
        if params['position_size'] <= 0:
            logger.error("Invalid position size")
            return False
        
        # Validate price
        if params['entry_price'] <= 0:
            logger.error("Invalid entry price")
            return False
        
        # Validate risk params
        risk_params = params['risk_params']
        if 'stop_loss_price' in risk_params:
            if risk_params['stop_loss_price'] >= params['entry_price']:
                logger.error("Invalid stop loss price")
                return False
        
        return True
    
    async def execute_strategy(self, execution_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute selected strategy with monitoring"""
        strategy_name = execution_params['strategy_name']
        
        try:
            # Create strategy state
            state = LiveStrategyState(
                strategy_name=strategy_name,
                entry_time=datetime.now(timezone.utc),
                entry_price=execution_params['entry_price'],
                current_price=execution_params['entry_price'],
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                position_size=execution_params['position_size'],
                stop_loss=execution_params['risk_params'].get('stop_loss_price', 0),
                take_profit=execution_params['risk_params'].get('take_profit_price', 0),
                signals=set(execution_params.get('signals', []))
            )
            
            # Register active strategy
            with self.lock:
                self.active_strategies[strategy_name] = state
            
            # Start monitoring
            if not self.monitoring_task:
                self.monitoring_task = asyncio.create_task(self._monitor_strategies())
            
            # Log execution
            logger.info(
                f"Executing {strategy_name}: "
                f"size=${state.position_size:.2f} "
                f"entry=${state.entry_price:.2f}"
            )
            
            # Record metrics
            self.metrics.record_strategy_execution(execution_params)
            
            return {
                'status': 'executed',
                'strategy_name': strategy_name,
                'entry_price': state.entry_price,
                'position_size': state.position_size,
                'risk_params': execution_params['risk_params']
            }
            
        except Exception as e:
            logger.error(f"Strategy execution error: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    async def _monitor_strategies(self):
        """Monitor active strategies in real-time"""
        while self.active_strategies:
            try:
                for strategy_name, state in list(self.active_strategies.items()):
                    # Check exit conditions
                    should_exit, reason = await self._check_exit_conditions(state)
                    
                    if should_exit:
                        await self._exit_strategy(strategy_name, reason)
                    else:
                        # Update trailing stop if needed
                        self._update_trailing_stop(state)
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _check_exit_conditions(
        self,
        state: LiveStrategyState
    ) -> Tuple[bool, str]:
        """Check if strategy should exit"""
        current_price = state.current_price
        
        # Stop loss
        if state.stop_loss > 0 and current_price <= state.stop_loss:
            return True, "stop_loss"
        
        # Take profit
        if state.take_profit > 0 and current_price >= state.take_profit:
            return True, "take_profit"
        
        # Trailing stop
        if state.trailing_stop is not None:
            if current_price <= state.highest_price * (1 - state.trailing_stop):
                return True, "trailing_stop"
        
        # Time limit
        elapsed = (datetime.now(timezone.utc) - state.entry_time).total_seconds()
        strategy_config = self.strategies.get(state.strategy_name, {})
        if elapsed > strategy_config.get('timeout', self.strategy_timeout):
            return True, "timeout"
        
        # Strategy-specific exits
        if state.strategy_name in ['funding_arbitrage', 'cross_exchange_arb']:
            # Check if arbitrage opportunity still exists
            if abs(state.unrealized_pnl / state.position_size) < 0.001:  # Less than 0.1%
                return True, "opportunity_closed"
        
        return False, ""
    
    def _update_trailing_stop(self, state: LiveStrategyState):
        """Update trailing stop if applicable"""
        if state.trailing_stop is None:
            return
        
        strategy_config = self.strategies.get(state.strategy_name, {})
        risk_params = strategy_config.get('risk_params', {})
        
        if 'trailing_stop' in risk_params:
            trailing_pct = risk_params['trailing_stop']
            
            # Activate trailing stop if profit threshold reached
            if not hasattr(state, 'trailing_activated'):
                profit_pct = (state.current_price - state.entry_price) / state.entry_price
                if profit_pct >= trailing_pct:
                    state.trailing_activated = True
                    state.trailing_stop = trailing_pct
            
            # Update stop level
            if hasattr(state, 'trailing_activated') and state.trailing_activated:
                new_stop = state.highest_price * (1 - trailing_pct)
                state.stop_loss = max(state.stop_loss, new_stop)
    
    async def _exit_strategy(self, strategy_name: str, reason: str):
        """Exit a strategy position"""
        with self.lock:
            if strategy_name not in self.active_strategies:
                return
            
            state = self.active_strategies[strategy_name]
            
            # Calculate final P&L
            state.realized_pnl = (state.current_price - state.entry_price) * state.position_size
            
            # Update performance
            self._update_performance(strategy_name, state, reason)
            
            # Remove from active
            del self.active_strategies[strategy_name]
            
            logger.info(
                f"Exited {strategy_name}: "
                f"reason={reason} "
                f"pnl=${state.realized_pnl:.2f} "
                f"return={state.realized_pnl/state.position_size:.2%}"
            )
    
    def _update_performance(self, strategy_name: str, state: LiveStrategyState, exit_reason: str):
        """Update strategy performance metrics"""
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = StrategyPerformance()
        
        perf = self.strategy_performance[strategy_name]
        
        # Update trade counts
        perf.total_trades += 1
        if state.realized_pnl > 0:
            perf.winning_trades += 1
            perf.avg_win = ((perf.avg_win * (perf.winning_trades - 1)) + state.realized_pnl) / perf.winning_trades
        else:
            perf.losing_trades += 1
            perf.avg_loss = ((perf.avg_loss * (perf.losing_trades - 1)) + abs(state.realized_pnl)) / perf.losing_trades
        
        # Update metrics
        perf.total_pnl += state.realized_pnl
        perf.win_rate = perf.winning_trades / perf.total_trades
        
        # Update daily returns
        return_pct = state.realized_pnl / state.position_size
        perf.daily_returns.append(return_pct)
        
        # Calculate Sharpe ratio (simplified)
        if len(perf.daily_returns) >= 20:
            returns = np.array(perf.daily_returns)
            perf.sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
        
        # Calculate profit factor
        if perf.avg_loss > 0:
            perf.profit_factor = perf.avg_win / perf.avg_loss
        
        # Update max drawdown
        perf.max_drawdown = min(perf.max_drawdown, state.unrealized_pnl / state.position_size)
        
        perf.last_updated = datetime.now(timezone.utc)
        
        # Save performance
        self._save_performance_data()
    
    def _calculate_daily_pnl(self) -> float:
        """Calculate today's P&L"""
        today = datetime.now(timezone.utc).date()
        daily_pnl = 0.0
        
        # Active positions unrealized P&L
        for state in self.active_strategies.values():
            daily_pnl += state.unrealized_pnl
        
        # Closed positions realized P&L (today only)
        for strategy_name, perf in self.strategy_performance.items():
            if perf.last_updated.date() == today:
                # This is approximate - would need trade history for exact calculation
                daily_pnl += perf.total_pnl / perf.total_trades if perf.total_trades > 0 else 0
        
        return daily_pnl
    
    def _extract_signals(self, analysis_bundle: Dict[str, Any]) -> List[str]:
        """Extract trading signals from analysis"""
        signals = []
        
        # Technical signals
        if 'technical' in analysis_bundle:
            tech = analysis_bundle['technical']
            if tech.get('trend') == 'bullish':
                signals.append('tech_bullish')
            elif tech.get('trend') == 'bearish':
                signals.append('tech_bearish')
        
        # Pattern signals
        if 'patterns' in analysis_bundle:
            patterns = analysis_bundle['patterns']
            if patterns.get('patterns'):
                signals.extend([p['name'] for p in patterns['patterns'][:3]])
        
        # Market signals
        if 'market' in analysis_bundle:
            market = analysis_bundle['market']
            if market.get('sentiment_score', 0) > 0.7:
                signals.append('strong_sentiment')
        
        return signals
    
    def _get_market_conditions(self, analysis_bundle: Dict[str, Any]) -> Dict[str, Any]:
        """Extract current market conditions"""
        return {
            'volatility': analysis_bundle.get('market', {}).get('volatility', 0),
            'regime': analysis_bundle.get('market', {}).get('regime', 'unknown'),
            'liquidity': analysis_bundle.get('orderbook', {}).get('liquidity_score', 1.0),
            'trend': analysis_bundle.get('technical', {}).get('trend', 'neutral')
        }
    
    def _get_strategy_win_rate(self, strategy_type: StrategyType) -> float:
        """Get historical win rate for strategy type"""
        total_trades = 0
        total_wins = 0
        
        for name, perf in self.strategy_performance.items():
            if name in self.strategies:
                if self.strategies[name]['type'] == strategy_type:
                    total_trades += perf.total_trades
                    total_wins += perf.winning_trades
        
        if total_trades > 0:
            return total_wins / total_trades
        
        # Default win rates by type
        defaults = {
            StrategyType.ARBITRAGE: 0.75,
            StrategyType.MOMENTUM: 0.55,
            StrategyType.MEAN_REVERSION: 0.60,
            StrategyType.MARKET_MAKING: 0.65,
            StrategyType.VOLATILITY: 0.50
        }
        
        return defaults.get(strategy_type, 0.50)
    
    def _is_market_hours(self, symbol: str) -> bool:
        """Check if within trading hours for symbol"""
        # For crypto, always true
        if 'BTC' in symbol or 'ETH' in symbol:
            return True
        
        # For traditional markets, check hours
        current_hour = datetime.now(timezone.utc).hour
        
        # US market hours (UTC)
        if 'USD' in symbol:
            return 13 <= current_hour <= 21
        
        # European hours
        if 'EUR' in symbol:
            return 7 <= current_hour <= 16
        
        # Asian hours
        if 'JPY' in symbol or 'CNY' in symbol:
            return 0 <= current_hour <= 8
        
        return True
    
    def _save_performance_data(self):
        """Save performance data to disk"""
        try:
            data = {
                name: {
                    'total_trades': perf.total_trades,
                    'winning_trades': perf.winning_trades,
                    'total_pnl': perf.total_pnl,
                    'win_rate': perf.win_rate,
                    'sharpe_ratio': perf.sharpe_ratio,
                    'last_updated': perf.last_updated.isoformat()
                }
                for name, perf in self.strategy_performance.items()
            }
            
            with open('logs/strategy_performance_live.json', 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving performance data: {e}")
    
    def _load_performance_data(self):
        """Load historical performance data"""
        try:
            with open('logs/strategy_performance_live.json', 'r') as f:
                data = json.load(f)
                
            for name, perf_data in data.items():
                perf = StrategyPerformance()
                perf.total_trades = perf_data['total_trades']
                perf.winning_trades = perf_data['winning_trades']
                perf.total_pnl = perf_data['total_pnl']
                perf.win_rate = perf_data['win_rate']
                perf.sharpe_ratio = perf_data['sharpe_ratio']
                perf.last_updated = datetime.fromisoformat(perf_data['last_updated'])
                
                self.strategy_performance[name] = perf
                
            logger.info(f"Loaded performance for {len(self.strategy_performance)} strategies")
            
        except FileNotFoundError:
            logger.info("No historical performance data found")
        except Exception as e:
            logger.error(f"Error loading performance data: {e}")
    
    def get_active_strategies_summary(self) -> Dict[str, Any]:
        """Get summary of active strategies"""
        with self.lock:
            return {
                'active_count': len(self.active_strategies),
                'total_allocated': sum(s.position_size for s in self.active_strategies.values()),
                'total_unrealized_pnl': sum(s.unrealized_pnl for s in self.active_strategies.values()),
                'strategies': [
                    {
                        'name': name,
                        'entry_time': state.entry_time.isoformat(),
                        'position_size': state.position_size,
                        'unrealized_pnl': state.unrealized_pnl,
                        'return_pct': state.unrealized_pnl / state.position_size if state.position_size > 0 else 0
                    }
                    for name, state in self.active_strategies.items()
                ]
            }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'overall': {
                'total_strategies': len(self.strategy_performance),
                'active_strategies': len(self.active_strategies),
                'total_pnl': sum(p.total_pnl for p in self.strategy_performance.values()),
                'total_trades': sum(p.total_trades for p in self.strategy_performance.values()),
                'avg_win_rate': np.mean([p.win_rate for p in self.strategy_performance.values() if p.total_trades > 0])
            },
            'by_strategy': {},
            'by_type': defaultdict(lambda: {'trades': 0, 'pnl': 0, 'win_rate': 0})
        }
        
        # Individual strategy performance
        for name, perf in self.strategy_performance.items():
            if perf.total_trades > 0:
                report['by_strategy'][name] = {
                    'trades': perf.total_trades,
                    'pnl': perf.total_pnl,
                    'win_rate': perf.win_rate,
                    'sharpe_ratio': perf.sharpe_ratio,
                    'profit_factor': perf.profit_factor,
                    'avg_win': perf.avg_win,
                    'avg_loss': perf.avg_loss
                }
                
                # Aggregate by type
                if name in self.strategies:
                    strategy_type = self.strategies[name]['type'].value
                    report['by_type'][strategy_type]['trades'] += perf.total_trades
                    report['by_type'][strategy_type]['pnl'] += perf.total_pnl
        
        # Calculate win rates by type
        for strategy_type in report['by_type']:
            trades = report['by_type'][strategy_type]['trades']
            if trades > 0:
                wins = sum(
                    self.strategy_performance[name].winning_trades
                    for name in self.strategies
                    if name in self.strategy_performance and
                    self.strategies[name]['type'].value == strategy_type
                )
                report['by_type'][strategy_type]['win_rate'] = wins / trades
        
        return dict(report)
    
    async def shutdown(self):
        """Gracefully shutdown strategy manager"""
        logger.info("Shutting down Strategy Manager")
        
        # Close all active positions
        for strategy_name in list(self.active_strategies.keys()):
            await self._exit_strategy(strategy_name, "shutdown")
        
        # Cancel monitoring
        if self.monitoring_task:
            self.monitoring_task.cancel()
        
        # Save final performance data
        self._save_performance_data()
        
        logger.info("Strategy Manager shutdown complete")