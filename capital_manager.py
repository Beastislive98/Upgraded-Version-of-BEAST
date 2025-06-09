"""
BEAST Trading System - Capital Manager
Capital allocation, portfolio management, and fund tracking
Merged from: capital_manager.py, capital_allocator.py
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
from enum import Enum

from config.settings import config
from utils.logger import get_logger

logger = get_logger(__name__)

class AllocationStrategy(Enum):
    """Capital allocation strategies"""
    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_WEIGHTED = "volatility_weighted"
    PERFORMANCE_BASED = "performance_based"
    DYNAMIC = "dynamic"

@dataclass
class Position:
    """Represents an active position"""
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    current_price: float
    size: float  # Position size in base currency
    value: float  # Current value
    entry_time: datetime
    strategy: str
    stop_loss: float
    take_profit: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    peak_value: float = 0.0
    drawdown: float = 0.0
    hold_time: timedelta = field(default_factory=timedelta)
    
    def update_price(self, new_price: float):
        """Update position with new price"""
        self.current_price = new_price
        self.value = self.size * new_price
        
        # Calculate unrealized PnL
        if self.side == 'long':
            self.unrealized_pnl = (new_price - self.entry_price) * self.size
        else:  # short
            self.unrealized_pnl = (self.entry_price - new_price) * self.size
        
        # Track peak value and drawdown
        if self.value > self.peak_value:
            self.peak_value = self.value
        self.drawdown = (self.peak_value - self.value) / self.peak_value if self.peak_value > 0 else 0
        
        # Update hold time
        self.hold_time = datetime.now(timezone.utc) - self.entry_time

@dataclass
class CapitalAllocation:
    """Capital allocation decision"""
    symbol: str
    allocation_pct: float
    allocation_amount: float
    max_position_value: float
    strategy: AllocationStrategy
    risk_budget: float
    rationale: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class CapitalAllocator:
    """Intelligent capital allocation across strategies and assets"""
    
    def __init__(self, strategy: AllocationStrategy = AllocationStrategy.DYNAMIC):
        self.strategy = strategy
        self.allocation_history = deque(maxlen=100)
        self.performance_window = 30  # days for performance-based allocation
        
    def allocate(
        self,
        available_capital: float,
        opportunities: List[Dict[str, Any]],
        current_positions: Dict[str, Position],
        risk_budgets: Dict[str, float]
    ) -> List[CapitalAllocation]:
        """Allocate capital across opportunities"""
        
        if self.strategy == AllocationStrategy.EQUAL_WEIGHT:
            return self._equal_weight_allocation(available_capital, opportunities)
        elif self.strategy == AllocationStrategy.RISK_PARITY:
            return self._risk_parity_allocation(available_capital, opportunities, risk_budgets)
        elif self.strategy == AllocationStrategy.KELLY_CRITERION:
            return self._kelly_allocation(available_capital, opportunities)
        elif self.strategy == AllocationStrategy.VOLATILITY_WEIGHTED:
            return self._volatility_weighted_allocation(available_capital, opportunities)
        elif self.strategy == AllocationStrategy.PERFORMANCE_BASED:
            return self._performance_based_allocation(available_capital, opportunities)
        else:  # DYNAMIC
            return self._dynamic_allocation(available_capital, opportunities, current_positions, risk_budgets)
    
    def _equal_weight_allocation(
        self,
        available_capital: float,
        opportunities: List[Dict[str, Any]]
    ) -> List[CapitalAllocation]:
        """Simple equal weight allocation"""
        allocations = []
        
        if not opportunities:
            return allocations
        
        allocation_per_opportunity = available_capital / len(opportunities)
        
        for opp in opportunities:
            allocation = CapitalAllocation(
                symbol=opp['symbol'],
                allocation_pct=1.0 / len(opportunities),
                allocation_amount=allocation_per_opportunity,
                max_position_value=allocation_per_opportunity,
                strategy=AllocationStrategy.EQUAL_WEIGHT,
                risk_budget=opp.get('risk_budget', 0.02),
                rationale={'method': 'equal_weight'}
            )
            allocations.append(allocation)
        
        return allocations
    
    def _risk_parity_allocation(
        self,
        available_capital: float,
        opportunities: List[Dict[str, Any]],
        risk_budgets: Dict[str, float]
    ) -> List[CapitalAllocation]:
        """Risk parity allocation - equal risk contribution"""
        allocations = []
        
        # Calculate risk-adjusted allocations
        total_inv_risk = sum(1.0 / risk_budgets.get(opp['symbol'], 0.02) for opp in opportunities)
        
        for opp in opportunities:
            symbol = opp['symbol']
            risk = risk_budgets.get(symbol, 0.02)
            
            # Allocation inversely proportional to risk
            weight = (1.0 / risk) / total_inv_risk
            allocation_amount = available_capital * weight
            
            allocation = CapitalAllocation(
                symbol=symbol,
                allocation_pct=weight,
                allocation_amount=allocation_amount,
                max_position_value=allocation_amount,
                strategy=AllocationStrategy.RISK_PARITY,
                risk_budget=risk,
                rationale={
                    'method': 'risk_parity',
                    'risk_weight': 1.0 / risk,
                    'normalized_weight': weight
                }
            )
            allocations.append(allocation)
        
        return allocations
    
    def _dynamic_allocation(
        self,
        available_capital: float,
        opportunities: List[Dict[str, Any]],
        current_positions: Dict[str, Position],
        risk_budgets: Dict[str, float]
    ) -> List[CapitalAllocation]:
        """Dynamic allocation based on multiple factors"""
        allocations = []
        
        # Score each opportunity
        scored_opportunities = []
        for opp in opportunities:
            score = self._score_opportunity(opp, current_positions)
            scored_opportunities.append((opp, score))
        
        # Sort by score
        scored_opportunities.sort(key=lambda x: x[1], reverse=True)
        
        # Allocate based on scores with risk limits
        total_score = sum(score for _, score in scored_opportunities)
        remaining_capital = available_capital
        
        for opp, score in scored_opportunities:
            if remaining_capital <= 0:
                break
            
            symbol = opp['symbol']
            
            # Base allocation by score
            base_weight = score / total_score if total_score > 0 else 0
            base_amount = available_capital * base_weight
            
            # Apply risk limits
            risk_limit = risk_budgets.get(symbol, 0.02) * available_capital
            allocation_amount = min(base_amount, risk_limit, remaining_capital)
            
            # Minimum allocation threshold
            if allocation_amount < available_capital * 0.01:  # Less than 1%
                continue
            
            allocation = CapitalAllocation(
                symbol=symbol,
                allocation_pct=allocation_amount / available_capital,
                allocation_amount=allocation_amount,
                max_position_value=allocation_amount * 1.2,  # 20% buffer
                strategy=AllocationStrategy.DYNAMIC,
                risk_budget=risk_budgets.get(symbol, 0.02),
                rationale={
                    'method': 'dynamic',
                    'opportunity_score': score,
                    'base_weight': base_weight,
                    'risk_limited': base_amount > allocation_amount
                }
            )
            allocations.append(allocation)
            remaining_capital -= allocation_amount
        
        return allocations
    
    def _score_opportunity(
        self,
        opportunity: Dict[str, Any],
        current_positions: Dict[str, Position]
    ) -> float:
        """Score an opportunity for dynamic allocation"""
        score = 0.0
        
        # Factor 1: Confidence (40% weight)
        confidence = opportunity.get('confidence', 0.5)
        score += confidence * 0.4
        
        # Factor 2: Risk-adjusted return (30% weight)
        expected_return = opportunity.get('expected_return', 0.05)
        risk = opportunity.get('risk_score', 0.5)
        risk_adjusted_return = expected_return / (risk + 0.1)  # Avoid division by zero
        score += min(risk_adjusted_return * 0.3, 0.3)  # Cap contribution
        
        # Factor 3: Strategy diversity (20% weight)
        strategy = opportunity.get('strategy', 'unknown')
        current_strategies = [pos.strategy for pos in current_positions.values()]
        if strategy not in current_strategies:
            score += 0.2  # Bonus for diversification
        
        # Factor 4: Market conditions alignment (10% weight)
        market_alignment = opportunity.get('market_alignment', 0.5)
        score += market_alignment * 0.1
        
        return score

class CapitalManager:
    """
    Comprehensive capital and portfolio management
    Tracks funds, positions, and performance
    """
    
    def __init__(self, config):
        self.config = config
        
        # Capital tracking
        self.total_capital = 100000.0  # Default starting capital
        self.available_capital = self.total_capital
        self.reserved_capital = 0.0
        self.margin_used = 0.0
        
        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.position_history = deque(maxlen=1000)
        self.closed_positions = deque(maxlen=100)
        
        # Performance tracking
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.peak_capital = self.total_capital
        self.max_drawdown = 0.0
        self.daily_pnl_history = deque(maxlen=365)
        self.monthly_returns = defaultdict(float)
        
        # Capital allocator
        self.allocator = CapitalAllocator(AllocationStrategy.DYNAMIC)
        
        # Risk budgets per symbol
        self.risk_budgets = {}
        
        # Transaction history
        self.transactions = deque(maxlen=1000)
        
        # Initialize from config
        self._initialize_from_config()
        
        logger.info(f"CapitalManager initialized with ${self.total_capital:,.2f}")
    
    def _initialize_from_config(self):
        """Initialize from configuration"""
        if hasattr(self.config, 'backtest'):
            self.total_capital = self.config.backtest.initial_capital
            self.available_capital = self.total_capital
    
    async def check_capital_available(
        self,
        required_amount: float,
        symbol: str
    ) -> Tuple[bool, Optional[str]]:
        """Check if sufficient capital is available"""
        # Account for margin if applicable
        margin_requirement = self._calculate_margin_requirement(symbol, required_amount)
        
        if margin_requirement > self.available_capital:
            return False, f"Insufficient capital: need ${margin_requirement:,.2f}, have ${self.available_capital:,.2f}"
        
        # Check position limits
        if len(self.positions) >= 10:
            return False, "Maximum position count reached"
        
        # Check concentration limits
        if symbol in self.positions:
            current_value = self.positions[symbol].value
            if (current_value + required_amount) > self.total_capital * 0.25:
                return False, f"Position concentration limit exceeded for {symbol}"
        
        return True, None
    
    async def allocate_capital(
        self,
        opportunities: List[Dict[str, Any]]
    ) -> List[CapitalAllocation]:
        """Allocate capital across opportunities"""
        # Update risk budgets
        for opp in opportunities:
            symbol = opp['symbol']
            risk_score = opp.get('risk_score', 0.5)
            
            # Dynamic risk budget based on risk score
            if risk_score < 0.3:
                self.risk_budgets[symbol] = 0.03  # 3% for low risk
            elif risk_score < 0.6:
                self.risk_budgets[symbol] = 0.02  # 2% for medium risk
            else:
                self.risk_budgets[symbol] = 0.01  # 1% for high risk
        
        # Get allocations
        allocations = self.allocator.allocate(
            self.available_capital,
            opportunities,
            self.positions,
            self.risk_budgets
        )
        
        # Record allocations
        for allocation in allocations:
            self.allocator.allocation_history.append(allocation)
        
        return allocations
    
    async def open_position(
        self,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
        strategy: str,
        stop_loss: float,
        take_profit: float
    ) -> Position:
        """Open a new position"""
        # Calculate position value
        position_value = size * entry_price
        
        # Check capital
        can_open, reason = await self.check_capital_available(position_value, symbol)
        if not can_open:
            raise ValueError(reason)
        
        # Create position
        position = Position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            current_price=entry_price,
            size=size,
            value=position_value,
            entry_time=datetime.now(timezone.utc),
            strategy=strategy,
            stop_loss=stop_loss,
            take_profit=take_profit,
            peak_value=position_value
        )
        
        # Update capital
        margin_used = self._calculate_margin_requirement(symbol, position_value)
        self.available_capital -= margin_used
        self.margin_used += margin_used
        
        # Store position
        self.positions[symbol] = position
        
        # Record transaction
        self._record_transaction({
            'type': 'open_position',
            'symbol': symbol,
            'side': side,
            'size': size,
            'price': entry_price,
            'value': position_value,
            'margin_used': margin_used,
            'timestamp': position.entry_time
        })
        
        logger.info(
            f"Opened {side} position: {symbol} "
            f"size={size} @ ${entry_price:,.2f}"
        )
        
        return position
    
    async def close_position(
        self,
        symbol: str,
        close_price: float,
        reason: str = "manual"
    ) -> Dict[str, Any]:
        """Close an existing position"""
        if symbol not in self.positions:
            raise ValueError(f"No position found for {symbol}")
        
        position = self.positions[symbol]
        
        # Update final price
        position.update_price(close_price)
        
        # Calculate realized PnL
        if position.side == 'long':
            pnl = (close_price - position.entry_price) * position.size
        else:  # short
            pnl = (position.entry_price - close_price) * position.size
        
        position.realized_pnl = pnl
        
        # Update capital
        self.realized_pnl += pnl
        self.total_capital += pnl
        
        # Release margin
        margin_released = self._calculate_margin_requirement(symbol, position.value)
        self.available_capital += margin_released + pnl
        self.margin_used -= margin_released
        
        # Move to closed positions
        self.closed_positions.append(position)
        del self.positions[symbol]
        
        # Record transaction
        close_data = {
            'type': 'close_position',
            'symbol': symbol,
            'side': position.side,
            'size': position.size,
            'entry_price': position.entry_price,
            'close_price': close_price,
            'pnl': pnl,
            'return_pct': pnl / (position.size * position.entry_price),
            'hold_time': position.hold_time.total_seconds() / 3600,  # hours
            'reason': reason,
            'timestamp': datetime.now(timezone.utc)
        }
        self._record_transaction(close_data)
        
        # Update daily PnL
        self._update_daily_pnl(pnl)
        
        logger.info(
            f"Closed {position.side} position: {symbol} "
            f"PnL=${pnl:,.2f} ({close_data['return_pct']:.2%})"
        )
        
        return close_data
    
    async def update_positions(self, price_updates: Dict[str, float]):
        """Update all positions with new prices"""
        self.unrealized_pnl = 0.0
        
        for symbol, position in self.positions.items():
            if symbol in price_updates:
                position.update_price(price_updates[symbol])
                self.unrealized_pnl += position.unrealized_pnl
        
        # Update drawdown
        current_total = self.total_capital + self.unrealized_pnl
        if current_total > self.peak_capital:
            self.peak_capital = current_total
        
        self.max_drawdown = max(
            self.max_drawdown,
            (self.peak_capital - current_total) / self.peak_capital if self.peak_capital > 0 else 0
        )
    
    def _calculate_margin_requirement(self, symbol: str, position_value: float) -> float:
        """Calculate margin requirement for position"""
        # Simplified margin calculation
        # In production, this would consider leverage, asset type, etc.
        
        leverage = 1.0  # No leverage by default
        
        # Crypto typically allows higher leverage
        if any(crypto in symbol for crypto in ['BTC', 'ETH', 'BNB']):
            leverage = 10.0
        
        return position_value / leverage
    
    def _record_transaction(self, transaction: Dict[str, Any]):
        """Record transaction for audit trail"""
        self.transactions.append(transaction)
        
        # Also save to position history if it's a close
        if transaction['type'] == 'close_position':
            self.position_history.append(transaction)
    
    def _update_daily_pnl(self, pnl: float):
        """Update daily PnL tracking"""
        today = datetime.now(timezone.utc).date()
        
        # Find or create today's entry
        if self.daily_pnl_history and self.daily_pnl_history[-1]['date'] == today:
            self.daily_pnl_history[-1]['pnl'] += pnl
        else:
            self.daily_pnl_history.append({
                'date': today,
                'pnl': pnl,
                'capital': self.total_capital
            })
        
        # Update monthly returns
        month_key = today.strftime('%Y-%m')
        self.monthly_returns[month_key] += pnl
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        total_value = self.total_capital + self.unrealized_pnl
        
        # Calculate position metrics
        position_values = [pos.value for pos in self.positions.values()]
        largest_position = max(position_values) if position_values else 0
        
        # Win rate calculation
        closed_trades = list(self.closed_positions)
        if closed_trades:
            winning_trades = sum(1 for pos in closed_trades if pos.realized_pnl > 0)
            win_rate = winning_trades / len(closed_trades)
            avg_win = np.mean([pos.realized_pnl for pos in closed_trades if pos.realized_pnl > 0]) if winning_trades > 0 else 0
            avg_loss = np.mean([pos.realized_pnl for pos in closed_trades if pos.realized_pnl <= 0]) if winning_trades < len(closed_trades) else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
        
        return {
            'total_capital': self.total_capital,
            'available_capital': self.available_capital,
            'margin_used': self.margin_used,
            'total_value': total_value,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_pnl': self.realized_pnl + self.unrealized_pnl,
            'return_pct': (total_value - 100000) / 100000,  # Assuming 100k start
            'max_drawdown': self.max_drawdown,
            'active_positions': len(self.positions),
            'largest_position_pct': largest_position / total_value if total_value > 0 else 0,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            'sharpe_ratio': self._calculate_sharpe_ratio()
        }
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from daily returns"""
        if len(self.daily_pnl_history) < 30:
            return 0.0
        
        # Calculate daily returns
        daily_returns = []
        for i in range(1, len(self.daily_pnl_history)):
            prev_capital = self.daily_pnl_history[i-1]['capital']
            curr_capital = self.daily_pnl_history[i]['capital']
            if prev_capital > 0:
                daily_return = (curr_capital - prev_capital) / prev_capital
                daily_returns.append(daily_return)
        
        if not daily_returns:
            return 0.0
        
        # Annualized Sharpe ratio
        avg_return = np.mean(daily_returns)
        std_return = np.std(daily_returns)
        
        if std_return == 0:
            return 0.0
        
        # Assuming 252 trading days
        sharpe = (avg_return * 252) / (std_return * np.sqrt(252))
        
        return sharpe
    
    def get_position_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a position"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        return {
            'symbol': position.symbol,
            'side': position.side,
            'entry_price': position.entry_price,
            'current_price': position.current_price,
            'size': position.size,
            'value': position.value,
            'unrealized_pnl': position.unrealized_pnl,
            'unrealized_pnl_pct': position.unrealized_pnl / (position.size * position.entry_price),
            'hold_time_hours': position.hold_time.total_seconds() / 3600,
            'drawdown': position.drawdown,
            'stop_loss': position.stop_loss,
            'take_profit': position.take_profit,
            'distance_to_stop': abs(position.current_price - position.stop_loss) / position.current_price,
            'distance_to_target': abs(position.take_profit - position.current_price) / position.current_price
        }
    
    def should_reduce_position(self, symbol: str) -> Tuple[bool, str]:
        """Check if position should be reduced"""
        if symbol not in self.positions:
            return False, "No position"
        
        position = self.positions[symbol]
        
        # Check drawdown
        if position.drawdown > 0.1:  # 10% drawdown
            return True, f"High drawdown: {position.drawdown:.2%}"
        
        # Check hold time
        if position.hold_time.days > 7:  # Held for more than a week
            if position.unrealized_pnl < 0:
                return True, "Long-held losing position"
        
        # Check portfolio concentration
        portfolio_value = self.total_capital + self.unrealized_pnl
        if position.value / portfolio_value > 0.25:
            return True, "Position too large relative to portfolio"
        
        return False, "OK"
    
    def export_performance_report(self, filepath: str):
        """Export detailed performance report"""
        summary = self.get_portfolio_summary()
        
        # Add detailed metrics
        report = {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'summary': summary,
            'positions': {
                symbol: self.get_position_details(symbol)
                for symbol in self.positions
            },
            'closed_trades': [
                {
                    'symbol': pos.symbol,
                    'side': pos.side,
                    'entry_price': pos.entry_price,
                    'exit_price': pos.current_price,
                    'pnl': pos.realized_pnl,
                    'hold_time_hours': pos.hold_time.total_seconds() / 3600,
                    'strategy': pos.strategy
                }
                for pos in list(self.closed_positions)[-20:]  # Last 20
            ],
            'monthly_returns': dict(self.monthly_returns),
            'capital_allocation': {
                'strategy': self.allocator.strategy.value,
                'risk_budgets': self.risk_budgets,
                'recent_allocations': [
                    {
                        'symbol': alloc.symbol,
                        'allocation_pct': alloc.allocation_pct,
                        'rationale': alloc.rationale
                    }
                    for alloc in list(self.allocator.allocation_history)[-10:]
                ]
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Performance report exported to {filepath}")
    
    def is_healthy(self) -> bool:
        """Check if capital manager is healthy"""
        # Check for major issues
        if self.available_capital < 0:
            return False
        
        if self.max_drawdown > 0.25:  # 25% drawdown
            return False
        
        if self.margin_used > self.total_capital:
            return False
        
        return True