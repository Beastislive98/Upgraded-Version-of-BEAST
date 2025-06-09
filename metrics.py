"""
BEAST Trading System - Metrics
Performance tracking, monitoring, and analytics
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
import threading
import asyncio
import aiohttp
from prometheus_client import Counter, Gauge, Histogram, Summary, Info

from config.settings import config

logger = logging.getLogger(__name__)

# Prometheus metrics
TRADE_COUNTER = Counter('beast_trades_total', 'Total number of trades', ['symbol', 'strategy', 'result'])
DECISION_COUNTER = Counter('beast_decisions_total', 'Total number of decisions', ['decision_type'])
CONFIDENCE_HISTOGRAM = Histogram('beast_confidence_score', 'Confidence score distribution', 
                                buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
PROFIT_GAUGE = Gauge('beast_total_profit', 'Total profit/loss')
WIN_RATE_GAUGE = Gauge('beast_win_rate', 'Current win rate')
LATENCY_SUMMARY = Summary('beast_operation_latency_seconds', 'Operation latency', ['operation'])
ERROR_COUNTER = Counter('beast_errors_total', 'Total number of errors', ['component', 'error_type'])
ACTIVE_POSITIONS_GAUGE = Gauge('beast_active_positions', 'Number of active positions')

@dataclass
class TradeMetrics:
    """Metrics for a single trade"""
    trade_id: str
    symbol: str
    strategy: str
    entry_time: datetime
    exit_time: Optional[datetime] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: float = 0.0
    side: str = ""
    profit_loss: float = 0.0
    profit_loss_pct: float = 0.0
    commission: float = 0.0
    is_winner: bool = False
    hold_time_seconds: float = 0.0
    max_drawdown: float = 0.0
    risk_reward_actual: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SessionMetrics:
    """Metrics for a trading session"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit_loss: float = 0.0
    total_commission: float = 0.0
    max_drawdown: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    average_win: float = 0.0
    average_loss: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    trades_by_strategy: Dict[str, int] = field(default_factory=dict)
    profit_by_strategy: Dict[str, float] = field(default_factory=dict)

class MetricsCollector:
    """
    Collects and aggregates trading system metrics
    """
    
    def __init__(self):
        self.current_session = SessionMetrics(
            session_id=f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            start_time=datetime.now(timezone.utc)
        )
        
        # Trade tracking
        self.trades: Dict[str, TradeMetrics] = {}
        self.active_trades: Dict[str, TradeMetrics] = {}
        
        # Performance tracking
        self.equity_curve = deque(maxlen=10000)
        self.returns = deque(maxlen=1000)
        
        # Decision tracking
        self.decision_history = deque(maxlen=1000)
        self.confidence_scores = deque(maxlen=1000)
        
        # Error tracking
        self.error_log = deque(maxlen=100)
        
        # Real-time metrics
        self.real_time_metrics = {
            'last_update': datetime.now(timezone.utc),
            'current_equity': 0.0,
            'daily_pnl': 0.0,
            'open_positions': 0,
            'pending_orders': 0
        }
        
        # Performance by time
        self.hourly_performance = defaultdict(lambda: {'trades': 0, 'pnl': 0.0})
        self.daily_performance = defaultdict(lambda: {'trades': 0, 'pnl': 0.0})
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        logger.info(f"MetricsCollector initialized - Session: {self.current_session.session_id}")
    
    def record_trade_entry(self, trade_data: Dict[str, Any]) -> str:
        """Record trade entry"""
        with self._lock:
            trade_id = trade_data.get('trade_id', f"trade_{len(self.trades) + 1}")
            
            trade = TradeMetrics(
                trade_id=trade_id,
                symbol=trade_data['symbol'],
                strategy=trade_data['strategy'],
                entry_time=datetime.now(timezone.utc),
                entry_price=float(trade_data['entry_price']),
                quantity=float(trade_data['quantity']),
                side=trade_data['side'],
                metadata=trade_data.get('metadata', {})
            )
            
            self.active_trades[trade_id] = trade
            self.current_session.total_trades += 1
            
            # Update strategy counts
            self.current_session.trades_by_strategy[trade.strategy] = \
                self.current_session.trades_by_strategy.get(trade.strategy, 0) + 1
            
            # Update Prometheus metrics
            ACTIVE_POSITIONS_GAUGE.set(len(self.active_trades))
            
            logger.info(f"Trade entry recorded: {trade_id} - {trade.symbol} {trade.side}")
            
            return trade_id
    
    def record_trade_exit(self, trade_id: str, exit_data: Dict[str, Any]):
        """Record trade exit"""
        with self._lock:
            if trade_id not in self.active_trades:
                logger.error(f"Trade not found: {trade_id}")
                return
            
            trade = self.active_trades[trade_id]
            trade.exit_time = datetime.now(timezone.utc)
            trade.exit_price = float(exit_data['exit_price'])
            trade.commission = float(exit_data.get('commission', 0))
            
            # Calculate P&L
            if trade.side == 'BUY':
                trade.profit_loss = (trade.exit_price - trade.entry_price) * trade.quantity - trade.commission
            else:  # SELL
                trade.profit_loss = (trade.entry_price - trade.exit_price) * trade.quantity - trade.commission
            
            trade.profit_loss_pct = trade.profit_loss / (trade.entry_price * trade.quantity) * 100
            trade.is_winner = trade.profit_loss > 0
            
            # Calculate hold time
            trade.hold_time_seconds = (trade.exit_time - trade.entry_time).total_seconds()
            
            # Update session metrics
            self.current_session.total_profit_loss += trade.profit_loss
            self.current_session.total_commission += trade.commission
            
            if trade.is_winner:
                self.current_session.winning_trades += 1
            else:
                self.current_session.losing_trades += 1
            
            # Update strategy profit
            self.current_session.profit_by_strategy[trade.strategy] = \
                self.current_session.profit_by_strategy.get(trade.strategy, 0) + trade.profit_loss
            
            # Move to completed trades
            self.trades[trade_id] = trade
            del self.active_trades[trade_id]
            
            # Update Prometheus metrics
            TRADE_COUNTER.labels(
                symbol=trade.symbol,
                strategy=trade.strategy,
                result='win' if trade.is_winner else 'loss'
            ).inc()
            PROFIT_GAUGE.set(self.current_session.total_profit_loss)
            ACTIVE_POSITIONS_GAUGE.set(len(self.active_trades))
            
            # Update equity curve
            self._update_equity_curve()
            
            # Update performance metrics
            self._update_performance_metrics()
            
            logger.info(
                f"Trade exit recorded: {trade_id} - "
                f"P&L: {trade.profit_loss:.2f} ({trade.profit_loss_pct:.2f}%)"
            )
    
    def record_decision(self, decision: Dict[str, Any]):
        """Record trading decision"""
        with self._lock:
            decision_record = {
                'timestamp': datetime.now(timezone.utc),
                'symbol': decision.get('symbol'),
                'decision': decision.get('decision'),
                'confidence': decision.get('confidence', 0),
                'reasoning': decision.get('reasoning', {})
            }
            
            self.decision_history.append(decision_record)
            
            # Track confidence scores
            confidence = decision.get('confidence', 0)
            self.confidence_scores.append(confidence)
            
            # Update Prometheus metrics
            DECISION_COUNTER.labels(decision_type=decision.get('decision', 'unknown')).inc()
            CONFIDENCE_HISTOGRAM.observe(confidence)
            
            # Log high-confidence no-trades for analysis
            if decision.get('decision') == 'NO_TRADE' and confidence > 0.5:
                logger.debug(
                    f"High confidence NO_TRADE: {decision.get('symbol')} - "
                    f"Confidence: {confidence:.2f}, Reason: {decision.get('reasoning')}"
                )
    
    def record_error(self, component: str, error: Exception, context: Dict[str, Any] = None):
        """Record system error"""
        with self._lock:
            error_record = {
                'timestamp': datetime.now(timezone.utc),
                'component': component,
                'error_type': type(error).__name__,
                'error_message': str(error),
                'context': context or {}
            }
            
            self.error_log.append(error_record)
            
            # Update Prometheus metrics
            ERROR_COUNTER.labels(
                component=component,
                error_type=type(error).__name__
            ).inc()
            
            logger.error(f"Error in {component}: {error}")
    
    def record_latency(self, operation: str, latency_seconds: float):
        """Record operation latency"""
        LATENCY_SUMMARY.labels(operation=operation).observe(latency_seconds)
    
    def _update_equity_curve(self):
        """Update equity curve"""
        current_equity = self.current_session.total_profit_loss
        self.equity_curve.append({
            'timestamp': datetime.now(timezone.utc),
            'equity': current_equity,
            'trades': self.current_session.total_trades
        })
        
        # Calculate returns
        if len(self.equity_curve) > 1:
            prev_equity = self.equity_curve[-2]['equity']
            if prev_equity != 0:
                return_pct = (current_equity - prev_equity) / abs(prev_equity) * 100
                self.returns.append(return_pct)
    
    def _update_performance_metrics(self):
        """Update performance metrics"""
        # Win rate
        total_closed = self.current_session.winning_trades + self.current_session.losing_trades
        if total_closed > 0:
            self.current_session.win_rate = self.current_session.winning_trades / total_closed
            WIN_RATE_GAUGE.set(self.current_session.win_rate)
        
        # Average win/loss
        if self.current_session.winning_trades > 0:
            total_wins = sum(t.profit_loss for t in self.trades.values() if t.is_winner)
            self.current_session.average_win = total_wins / self.current_session.winning_trades
        
        if self.current_session.losing_trades > 0:
            total_losses = sum(abs(t.profit_loss) for t in self.trades.values() if not t.is_winner)
            self.current_session.average_loss = total_losses / self.current_session.losing_trades
        
        # Profit factor
        if self.current_session.average_loss > 0:
            self.current_session.profit_factor = self.current_session.average_win / self.current_session.average_loss
        
        # Max drawdown
        if self.equity_curve:
            equity_values = [e['equity'] for e in self.equity_curve]
            self.current_session.max_drawdown = self._calculate_max_drawdown(equity_values)
        
        # Sharpe ratio (simplified daily)
        if len(self.returns) > 1:
            returns_array = np.array(list(self.returns))
            if returns_array.std() > 0:
                self.current_session.sharpe_ratio = (returns_array.mean() / returns_array.std()) * np.sqrt(252)
    
    def _calculate_max_drawdown(self, equity_values: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not equity_values:
            return 0.0
        
        peak = equity_values[0]
        max_dd = 0.0
        
        for value in equity_values:
            if value > peak:
                peak = value
            else:
                dd = (peak - value) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
        
        return max_dd
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current trading statistics"""
        with self._lock:
            # Calculate real-time metrics
            active_pnl = sum(
                self._calculate_unrealized_pnl(trade) 
                for trade in self.active_trades.values()
            )
            
            # Get today's performance
            today = datetime.now(timezone.utc).date()
            today_trades = [
                t for t in self.trades.values() 
                if t.entry_time.date() == today
            ]
            today_pnl = sum(t.profit_loss for t in today_trades)
            
            return {
                'session_id': self.current_session.session_id,
                'uptime_hours': (datetime.now(timezone.utc) - self.current_session.start_time).total_seconds() / 3600,
                'total_trades': self.current_session.total_trades,
                'active_trades': len(self.active_trades),
                'win_rate': self.current_session.win_rate,
                'total_pnl': self.current_session.total_profit_loss,
                'unrealized_pnl': active_pnl,
                'today_pnl': today_pnl,
                'today_trades': len(today_trades),
                'profit_factor': self.current_session.profit_factor,
                'sharpe_ratio': self.current_session.sharpe_ratio,
                'max_drawdown': self.current_session.max_drawdown,
                'average_confidence': np.mean(list(self.confidence_scores)) if self.confidence_scores else 0,
                'error_count': len(self.error_log)
            }
    
    def _calculate_unrealized_pnl(self, trade: TradeMetrics) -> float:
        """Calculate unrealized P&L for active trade"""
        # This would need current market price
        # Simplified for now
        return 0.0
    
    def get_strategy_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance breakdown by strategy"""
        with self._lock:
            strategy_stats = {}
            
            for strategy in self.current_session.trades_by_strategy:
                strategy_trades = [
                    t for t in self.trades.values() 
                    if t.strategy == strategy
                ]
                
                if strategy_trades:
                    wins = sum(1 for t in strategy_trades if t.is_winner)
                    total = len(strategy_trades)
                    total_pnl = sum(t.profit_loss for t in strategy_trades)
                    
                    strategy_stats[strategy] = {
                        'total_trades': total,
                        'wins': wins,
                        'losses': total - wins,
                        'win_rate': wins / total if total > 0 else 0,
                        'total_pnl': total_pnl,
                        'avg_pnl': total_pnl / total if total > 0 else 0,
                        'best_trade': max((t.profit_loss for t in strategy_trades), default=0),
                        'worst_trade': min((t.profit_loss for t in strategy_trades), default=0)
                    }
            
            return strategy_stats
    
    def get_time_analysis(self) -> Dict[str, Any]:
        """Analyze performance by time"""
        with self._lock:
            # Hourly analysis
            hourly_stats = {}
            for trade in self.trades.values():
                hour = trade.entry_time.hour
                if hour not in hourly_stats:
                    hourly_stats[hour] = {'trades': 0, 'pnl': 0.0, 'win_rate': 0.0}
                
                hourly_stats[hour]['trades'] += 1
                hourly_stats[hour]['pnl'] += trade.profit_loss
            
            # Calculate win rates per hour
            for hour in hourly_stats:
                hour_trades = [
                    t for t in self.trades.values() 
                    if t.entry_time.hour == hour
                ]
                wins = sum(1 for t in hour_trades if t.is_winner)
                hourly_stats[hour]['win_rate'] = wins / len(hour_trades) if hour_trades else 0
            
            # Day of week analysis
            dow_stats = defaultdict(lambda: {'trades': 0, 'pnl': 0.0})
            for trade in self.trades.values():
                dow = trade.entry_time.strftime('%A')
                dow_stats[dow]['trades'] += 1
                dow_stats[dow]['pnl'] += trade.profit_loss
            
            return {
                'hourly_performance': dict(hourly_stats),
                'day_of_week_performance': dict(dow_stats),
                'best_hour': max(hourly_stats.items(), key=lambda x: x[1]['pnl'])[0] if hourly_stats else None,
                'worst_hour': min(hourly_stats.items(), key=lambda x: x[1]['pnl'])[0] if hourly_stats else None
            }
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get risk-related metrics"""
        with self._lock:
            if not self.trades:
                return {}
            
            trade_returns = [t.profit_loss_pct for t in self.trades.values()]
            
            if not trade_returns:
                return {}
            
            returns_array = np.array(trade_returns)
            
            return {
                'value_at_risk_95': np.percentile(returns_array, 5),
                'expected_shortfall_95': returns_array[returns_array <= np.percentile(returns_array, 5)].mean(),
                'max_consecutive_losses': self._calculate_max_consecutive_losses(),
                'max_loss_streak_pnl': self._calculate_max_loss_streak_pnl(),
                'recovery_factor': self.current_session.total_profit_loss / self.current_session.max_drawdown 
                    if self.current_session.max_drawdown > 0 else 0,
                'risk_reward_ratio': self.current_session.average_win / self.current_session.average_loss 
                    if self.current_session.average_loss > 0 else 0
            }
    
    def _calculate_max_consecutive_losses(self) -> int:
        """Calculate maximum consecutive losses"""
        if not self.trades:
            return 0
        
        max_losses = 0
        current_losses = 0
        
        for trade in sorted(self.trades.values(), key=lambda x: x.exit_time or x.entry_time):
            if not trade.is_winner:
                current_losses += 1
                max_losses = max(max_losses, current_losses)
            else:
                current_losses = 0
        
        return max_losses
    
    def _calculate_max_loss_streak_pnl(self) -> float:
        """Calculate maximum loss during losing streak"""
        if not self.trades:
            return 0.0
        
        max_loss = 0.0
        current_loss = 0.0
        
        for trade in sorted(self.trades.values(), key=lambda x: x.exit_time or x.entry_time):
            if not trade.is_winner:
                current_loss += trade.profit_loss
                max_loss = min(max_loss, current_loss)
            else:
                current_loss = 0.0
        
        return abs(max_loss)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        return {
            'current_stats': self.get_current_stats(),
            'strategy_performance': self.get_strategy_performance(),
            'time_analysis': self.get_time_analysis(),
            'risk_metrics': self.get_risk_metrics(),
            'session_metrics': asdict(self.current_session),
            'recent_errors': list(self.error_log)[-10:] if self.error_log else []
        }
    
    def export_metrics(self, filepath: str):
        """Export metrics to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.get_summary(), f, indent=2, default=str)
            logger.info(f"Metrics exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
    
    def export_trades_csv(self, filepath: str):
        """Export trades to CSV"""
        try:
            if self.trades:
                trades_data = []
                for trade in self.trades.values():
                    trades_data.append({
                        'trade_id': trade.trade_id,
                        'symbol': trade.symbol,
                        'strategy': trade.strategy,
                        'entry_time': trade.entry_time,
                        'exit_time': trade.exit_time,
                        'side': trade.side,
                        'quantity': trade.quantity,
                        'entry_price': trade.entry_price,
                        'exit_price': trade.exit_price,
                        'profit_loss': trade.profit_loss,
                        'profit_loss_pct': trade.profit_loss_pct,
                        'commission': trade.commission,
                        'is_winner': trade.is_winner,
                        'hold_time_hours': trade.hold_time_seconds / 3600
                    })
                
                df = pd.DataFrame(trades_data)
                df.to_csv(filepath, index=False)
                logger.info(f"Trades exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export trades: {e}")
    
    async def send_metrics_webhook(self, webhook_url: str):
        """Send metrics summary to webhook"""
        if not webhook_url:
            return
        
        try:
            summary = self.get_summary()
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=summary) as response:
                    if response.status == 200:
                        logger.debug("Metrics sent to webhook successfully")
                    else:
                        logger.error(f"Webhook failed with status {response.status}")
        except Exception as e:
            logger.error(f"Failed to send metrics webhook: {e}")

# Global metrics instance
metrics_collector = MetricsCollector()

# Convenience functions
def record_trade_entry(trade_data: Dict[str, Any]) -> str:
    """Record trade entry"""
    return metrics_collector.record_trade_entry(trade_data)

def record_trade_exit(trade_id: str, exit_data: Dict[str, Any]):
    """Record trade exit"""
    metrics_collector.record_trade_exit(trade_id, exit_data)

def record_decision(decision: Dict[str, Any]):
    """Record trading decision"""
    metrics_collector.record_decision(decision)

def record_error(component: str, error: Exception, context: Dict[str, Any] = None):
    """Record system error"""
    metrics_collector.record_error(component, error, context)

def get_current_stats() -> Dict[str, Any]:
    """Get current statistics"""
    return metrics_collector.get_current_stats()

def get_metrics_summary() -> Dict[str, Any]:
    """Get comprehensive metrics summary"""
    return metrics_collector.get_summary()