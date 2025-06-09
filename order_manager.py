"""
BEAST Trading System - Order Manager
Order execution, monitoring, and lifecycle management
Merged from: order_execution.py, trade_executor.py, monitoring.py, dynamic_execution.py
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
import numpy as np
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import ccxt.async_support as ccxt
import json

from config.settings import config
from utils.logger import get_logger
from execution.capital_manager import Position

logger = get_logger(__name__)

class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class ExecutionStrategy(Enum):
    """Order execution strategies"""
    IMMEDIATE = "immediate"
    TWAP = "twap"  # Time-weighted average price
    VWAP = "vwap"  # Volume-weighted average price
    ICEBERG = "iceberg"  # Hidden quantity
    ADAPTIVE = "adaptive"  # Dynamic execution

@dataclass
class Order:
    """Represents a trading order"""
    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    type: OrderType
    quantity: float
    price: Optional[float]
    status: OrderStatus
    strategy: str
    execution_strategy: ExecutionStrategy
    created_at: datetime
    updated_at: datetime
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_fill(self, fill_qty: float, fill_price: float, fee: float = 0):
        """Update order with partial or full fill"""
        # Update average fill price
        total_value = (self.filled_quantity * self.avg_fill_price) + (fill_qty * fill_price)
        self.filled_quantity += fill_qty
        self.avg_fill_price = total_value / self.filled_quantity if self.filled_quantity > 0 else 0
        
        # Update fees
        self.fees += fee
        
        # Update status
        if self.filled_quantity >= self.quantity * 0.99:  # 99% filled considered complete
            self.status = OrderStatus.FILLED
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIAL
        
        # Calculate slippage for market orders
        if self.type == OrderType.MARKET and self.price:
            self.slippage = (self.avg_fill_price - self.price) / self.price
        
        self.updated_at = datetime.now(timezone.utc)

@dataclass
class ExecutionPlan:
    """Execution plan for complex orders"""
    total_quantity: float
    chunks: List[Dict[str, Any]]
    strategy: ExecutionStrategy
    time_horizon: timedelta
    price_limits: Dict[str, float]
    urgency: float  # 0-1 scale

class DynamicExecutor:
    """Dynamic order execution with smart routing"""
    
    def __init__(self):
        self.execution_history = deque(maxlen=100)
        self.market_impact_model = self._initialize_impact_model()
        
    def create_execution_plan(
        self,
        symbol: str,
        side: str,
        quantity: float,
        market_data: Dict[str, Any],
        urgency: float = 0.5
    ) -> ExecutionPlan:
        """Create optimal execution plan"""
        
        # Estimate market impact
        impact = self._estimate_market_impact(symbol, quantity, market_data)
        
        # Choose execution strategy based on order size and urgency
        if urgency > 0.8 or impact < 0.001:
            strategy = ExecutionStrategy.IMMEDIATE
            chunks = [{'quantity': quantity, 'delay': 0}]
            time_horizon = timedelta(minutes=1)
        elif quantity > market_data.get('avg_volume', 0) * 0.01:
            # Large order - use TWAP or Iceberg
            strategy = ExecutionStrategy.TWAP if urgency < 0.3 else ExecutionStrategy.ICEBERG
            chunks = self._create_twap_chunks(quantity, urgency)
            time_horizon = timedelta(minutes=30 if urgency < 0.3 else 10)
        else:
            # Adaptive execution
            strategy = ExecutionStrategy.ADAPTIVE
            chunks = self._create_adaptive_chunks(quantity, market_data, urgency)
            time_horizon = timedelta(minutes=15)
        
        # Set price limits based on strategy
        current_price = market_data.get('last_price', 0)
        if side == 'buy':
            price_limits = {
                'max': current_price * (1 + 0.001 * (1 + urgency)),  # More tolerance with urgency
                'target': current_price
            }
        else:
            price_limits = {
                'min': current_price * (1 - 0.001 * (1 + urgency)),
                'target': current_price
            }
        
        return ExecutionPlan(
            total_quantity=quantity,
            chunks=chunks,
            strategy=strategy,
            time_horizon=time_horizon,
            price_limits=price_limits,
            urgency=urgency
        )
    
    def _estimate_market_impact(
        self,
        symbol: str,
        quantity: float,
        market_data: Dict[str, Any]
    ) -> float:
        """Estimate market impact of order"""
        # Simplified market impact model
        avg_volume = market_data.get('avg_volume', 1000000)
        order_size_pct = quantity / avg_volume
        
        # Square-root model for market impact
        impact = 0.1 * np.sqrt(order_size_pct)
        
        # Adjust for liquidity
        spread = market_data.get('spread_pct', 0.1)
        impact *= (1 + spread)
        
        return min(impact, 0.05)  # Cap at 5%
    
    def _create_twap_chunks(self, quantity: float, urgency: float) -> List[Dict[str, Any]]:
        """Create TWAP execution chunks"""
        num_chunks = int(10 * (1 - urgency)) + 2  # 2-10 chunks based on urgency
        chunk_size = quantity / num_chunks
        
        chunks = []
        for i in range(num_chunks):
            chunks.append({
                'quantity': chunk_size,
                'delay': i * (300 / num_chunks),  # Spread over 5 minutes
                'type': 'limit'
            })
        
        return chunks
    
    def _create_adaptive_chunks(
        self,
        quantity: float,
        market_data: Dict[str, Any],
        urgency: float
    ) -> List[Dict[str, Any]]:
        """Create adaptive execution chunks based on market conditions"""
        volatility = market_data.get('volatility', 0.02)
        
        # Adjust chunk size based on volatility
        if volatility > 0.03:
            # High volatility - smaller chunks
            num_chunks = 5
        else:
            # Normal volatility
            num_chunks = 3
        
        # Create variable-sized chunks
        chunks = []
        remaining = quantity
        
        for i in range(num_chunks):
            # Front-load if urgent, back-load if patient
            if urgency > 0.5:
                chunk_pct = 0.5 if i == 0 else 0.3 if i == 1 else 0.2
            else:
                chunk_pct = 0.2 if i == 0 else 0.3 if i == 1 else 0.5
            
            chunk_size = min(quantity * chunk_pct, remaining)
            remaining -= chunk_size
            
            chunks.append({
                'quantity': chunk_size,
                'delay': i * 60,  # 1 minute between chunks
                'type': 'limit' if urgency < 0.7 else 'market'
            })
        
        return chunks
    
    def _initialize_impact_model(self) -> Dict[str, float]:
        """Initialize market impact model parameters"""
        return {
            'permanent_impact': 0.1,
            'temporary_impact': 0.05,
            'decay_rate': 0.8
        }

class OrderManager:
    """
    Comprehensive order execution and monitoring
    Handles order lifecycle from creation to completion
    """
    
    def __init__(self, config):
        self.config = config
        self.exchange = None
        self._init_exchange()
        
        # Order tracking
        self.active_orders: Dict[str, Order] = {}
        self.order_history = deque(maxlen=1000)
        self.position_orders: Dict[str, List[str]] = {}  # symbol -> order_ids
        
        # Execution components
        self.dynamic_executor = DynamicExecutor()
        self.execution_plans: Dict[str, ExecutionPlan] = {}
        
        # Monitoring
        self.fill_stream = asyncio.Queue()
        self.monitoring_task = None
        
        # Performance metrics
        self.execution_metrics = {
            'total_orders': 0,
            'filled_orders': 0,
            'rejected_orders': 0,
            'avg_slippage': 0.0,
            'total_fees': 0.0
        }
        
        logger.info("OrderManager initialized")
    
    def _init_exchange(self):
        """Initialize exchange connection"""
        try:
            exchange_class = getattr(ccxt, self.config.exchange.exchange_id)
            self.exchange = exchange_class({
                'apiKey': self.config.exchange.api_key,
                'secret': self.config.exchange.api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future' if 'future' in self.config.exchange.exchange_id else 'spot'
                }
            })
            
            if self.config.exchange.testnet:
                self.exchange.set_sandbox_mode(True)
                
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            raise
    
    async def execute(
        self,
        symbol: str,
        strategy: Dict[str, Any],
        risk_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a trading strategy
        Returns execution result
        """
        try:
            # Determine order parameters
            side = self._determine_side(strategy)
            quantity = self._calculate_quantity(
                symbol,
                strategy['position_size'],
                risk_params
            )
            
            # Get current market data
            market_data = await self._get_market_data(symbol)
            
            # Create execution plan
            urgency = self._calculate_urgency(strategy, market_data)
            execution_plan = self.dynamic_executor.create_execution_plan(
                symbol,
                side,
                quantity,
                market_data,
                urgency
            )
            
            # Store execution plan
            plan_id = f"{symbol}_{datetime.now().timestamp()}"
            self.execution_plans[plan_id] = execution_plan
            
            # Execute based on plan
            if execution_plan.strategy == ExecutionStrategy.IMMEDIATE:
                result = await self._execute_immediate(
                    symbol,
                    side,
                    quantity,
                    strategy,
                    market_data
                )
            else:
                result = await self._execute_complex(
                    symbol,
                    side,
                    execution_plan,
                    strategy,
                    market_data
                )
            
            # Set stop loss and take profit orders
            if result['status'] == 'success':
                await self._place_exit_orders(
                    symbol,
                    result['entry_price'],
                    result['size'],
                    risk_params
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Execution error for {symbol}: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'symbol': symbol
            }
    
    async def _execute_immediate(
        self,
        symbol: str,
        side: str,
        quantity: float,
        strategy: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute immediate market order"""
        # Create order
        order = Order(
            id=self._generate_order_id(),
            symbol=symbol,
            side=side,
            type=OrderType.MARKET,
            quantity=quantity,
            price=market_data['last_price'],
            status=OrderStatus.PENDING,
            strategy=strategy['name'],
            execution_strategy=ExecutionStrategy.IMMEDIATE,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            metadata={'strategy_params': strategy.get('parameters', {})}
        )
        
        # Submit order
        self.active_orders[order.id] = order
        
        try:
            # Place market order
            exchange_order = await self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=quantity
            )
            
            # Update order status
            order.status = OrderStatus.SUBMITTED
            order.metadata['exchange_order_id'] = exchange_order['id']
            
            # Wait for fill (with timeout)
            filled_order = await self._wait_for_fill(order.id, timeout=30)
            
            if filled_order.status == OrderStatus.FILLED:
                self.execution_metrics['filled_orders'] += 1
                
                return {
                    'status': 'success',
                    'order_id': order.id,
                    'symbol': symbol,
                    'side': side,
                    'size': filled_order.filled_quantity,
                    'entry_price': filled_order.avg_fill_price,
                    'fees': filled_order.fees,
                    'slippage': filled_order.slippage,
                    'execution_time': (filled_order.updated_at - filled_order.created_at).total_seconds()
                }
            else:
                return {
                    'status': 'partial',
                    'order_id': order.id,
                    'filled': filled_order.filled_quantity,
                    'remaining': quantity - filled_order.filled_quantity
                }
                
        except Exception as e:
            order.status = OrderStatus.REJECTED
            self.execution_metrics['rejected_orders'] += 1
            raise
    
    async def _execute_complex(
        self,
        symbol: str,
        side: str,
        execution_plan: ExecutionPlan,
        strategy: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute complex order with multiple chunks"""
        chunk_orders = []
        total_filled = 0
        total_value = 0
        total_fees = 0
        
        for i, chunk in enumerate(execution_plan.chunks):
            # Wait for delay
            if chunk['delay'] > 0:
                await asyncio.sleep(chunk['delay'])
            
            # Update market data
            market_data = await self._get_market_data(symbol)
            
            # Determine order type and price
            if chunk.get('type') == 'limit':
                order_type = OrderType.LIMIT
                if side == 'buy':
                    price = min(market_data['bid'] * 1.0001, execution_plan.price_limits['max'])
                else:
                    price = max(market_data['ask'] * 0.9999, execution_plan.price_limits.get('min', 0))
            else:
                order_type = OrderType.MARKET
                price = market_data['last_price']
            
            # Create chunk order
            order = Order(
                id=self._generate_order_id(),
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=chunk['quantity'],
                price=price,
                status=OrderStatus.PENDING,
                strategy=strategy['name'],
                execution_strategy=execution_plan.strategy,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                metadata={
                    'chunk': i + 1,
                    'total_chunks': len(execution_plan.chunks),
                    'parent_strategy': strategy.get('parameters', {})
                }
            )
            
            # Execute chunk
            self.active_orders[order.id] = order
            chunk_orders.append(order)
            
            try:
                if order_type == OrderType.MARKET:
                    exchange_order = await self.exchange.create_order(
                        symbol=symbol,
                        type='market',
                        side=side,
                        amount=chunk['quantity']
                    )
                else:
                    exchange_order = await self.exchange.create_order(
                        symbol=symbol,
                        type='limit',
                        side=side,
                        amount=chunk['quantity'],
                        price=price
                    )
                
                order.metadata['exchange_order_id'] = exchange_order['id']
                
                # Wait for fill
                filled_order = await self._wait_for_fill(order.id, timeout=60)
                
                if filled_order.filled_quantity > 0:
                    total_filled += filled_order.filled_quantity
                    total_value += filled_order.filled_quantity * filled_order.avg_fill_price
                    total_fees += filled_order.fees
                    
            except Exception as e:
                logger.error(f"Chunk {i+1} execution failed: {e}")
                continue
        
        # Calculate results
        if total_filled > 0:
            avg_price = total_value / total_filled
            
            return {
                'status': 'success' if total_filled >= execution_plan.total_quantity * 0.95 else 'partial',
                'order_ids': [o.id for o in chunk_orders],
                'symbol': symbol,
                'side': side,
                'size': total_filled,
                'entry_price': avg_price,
                'fees': total_fees,
                'slippage': (avg_price - market_data['last_price']) / market_data['last_price'],
                'execution_strategy': execution_plan.strategy.value,
                'chunks_executed': len(chunk_orders)
            }
        else:
            return {
                'status': 'failed',
                'reason': 'No chunks executed successfully'
            }
    
    async def _place_exit_orders(
        self,
        symbol: str,
        entry_price: float,
        position_size: float,
        risk_params: Dict[str, Any]
    ):
        """Place stop loss and take profit orders"""
        try:
            # Calculate exit prices
            stop_loss_price = entry_price * (1 - risk_params['stop_loss'])
            take_profit_price = entry_price * (1 + risk_params['take_profit'])
            
            # Place stop loss order
            sl_order = Order(
                id=self._generate_order_id(),
                symbol=symbol,
                side='sell',  # Opposite of entry
                type=OrderType.STOP_LOSS,
                quantity=position_size,
                price=stop_loss_price,
                status=OrderStatus.PENDING,
                strategy='stop_loss',
                execution_strategy=ExecutionStrategy.IMMEDIATE,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                metadata={'entry_price': entry_price, 'trigger': 'stop_loss'}
            )
            
            # Place take profit order
            tp_order = Order(
                id=self._generate_order_id(),
                symbol=symbol,
                side='sell',
                type=OrderType.TAKE_PROFIT,
                quantity=position_size,
                price=take_profit_price,
                status=OrderStatus.PENDING,
                strategy='take_profit',
                execution_strategy=ExecutionStrategy.IMMEDIATE,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                metadata={'entry_price': entry_price, 'trigger': 'take_profit'}
            )
            
            # Submit orders
            self.active_orders[sl_order.id] = sl_order
            self.active_orders[tp_order.id] = tp_order
            
            # Track position orders
            if symbol not in self.position_orders:
                self.position_orders[symbol] = []
            self.position_orders[symbol].extend([sl_order.id, tp_order.id])
            
            logger.info(f"Exit orders placed for {symbol}: SL=${stop_loss_price:.2f}, TP=${take_profit_price:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to place exit orders: {e}")
    
    async def close_position(self, symbol: str, position: Position) -> Dict[str, Any]:
        """Close a position with market order"""
        try:
            # Cancel any pending orders for this position
            await self._cancel_position_orders(symbol)
            
            # Determine close side (opposite of position)
            close_side = 'sell' if position.side == 'long' else 'buy'
            
            # Get current market data
            market_data = await self._get_market_data(symbol)
            
            # Execute close order
            result = await self._execute_immediate(
                symbol,
                close_side,
                position.size,
                {'name': 'position_close'},
                market_data
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to close position {symbol}: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def _cancel_position_orders(self, symbol: str):
        """Cancel all orders for a position"""
        if symbol in self.position_orders:
            for order_id in self.position_orders[symbol]:
                if order_id in self.active_orders:
                    order = self.active_orders[order_id]
                    if order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]:
                        try:
                            await self.exchange.cancel_order(
                                order.metadata.get('exchange_order_id'),
                                symbol
                            )
                            order.status = OrderStatus.CANCELLED
                        except Exception as e:
                            logger.error(f"Failed to cancel order {order_id}: {e}")
            
            # Clear position orders
            del self.position_orders[symbol]
    
    async def _wait_for_fill(self, order_id: str, timeout: int = 30) -> Order:
        """Wait for order to be filled"""
        start_time = datetime.now(timezone.utc)
        order = self.active_orders[order_id]
        
        while order.status not in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            # Check timeout
            if (datetime.now(timezone.utc) - start_time).total_seconds() > timeout:
                # Try to cancel order
                try:
                    await self.exchange.cancel_order(
                        order.metadata.get('exchange_order_id'),
                        order.symbol
                    )
                    order.status = OrderStatus.CANCELLED
                except:
                    pass
                break
            
            # Check order status
            try:
                exchange_order = await self.exchange.fetch_order(
                    order.metadata.get('exchange_order_id'),
                    order.symbol
                )
                
                # Update order based on exchange status
                if exchange_order['status'] == 'closed':
                    order.update_fill(
                        exchange_order['filled'],
                        exchange_order['average'],
                        exchange_order.get('fee', {}).get('cost', 0)
                    )
                elif exchange_order['status'] == 'canceled':
                    order.status = OrderStatus.CANCELLED
                    
            except Exception as e:
                logger.error(f"Error checking order status: {e}")
            
            await asyncio.sleep(1)
        
        # Move to history
        self.order_history.append(order)
        if order.status == OrderStatus.FILLED:
            del self.active_orders[order_id]
        
        return order
    
    def _determine_side(self, strategy: Dict[str, Any]) -> str:
        """Determine order side from strategy"""
        direction = strategy.get('direction', 'long')
        strategy_type = strategy.get('type', '')
        
        # Handle different strategy types
        if 'short' in strategy.get('name', '').lower() or direction == 'short':
            return 'sell'
        else:
            return 'buy'
    
    def _calculate_quantity(
        self,
        symbol: str,
        position_size: float,
        risk_params: Dict[str, Any]
    ) -> float:
        """Calculate order quantity from position size"""
        # This is simplified - in production would consider:
        # - Contract size
        # - Minimum order size
        # - Step size
        # - Available balance
        
        # For now, return position size directly
        # Assuming position_size is already in base currency units
        return position_size
    
    def _calculate_urgency(
        self,
        strategy: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> float:
        """Calculate execution urgency (0-1 scale)"""
        urgency = 0.5  # Default medium urgency
        
        # Increase urgency for momentum strategies
        if 'momentum' in strategy.get('name', '').lower():
            urgency += 0.2
        
        # Increase urgency if volatility is rising
        if market_data.get('volatility', 0) > 0.03:
            urgency += 0.1
        
        # Increase urgency for time-sensitive strategies
        if strategy.get('time_sensitive', False):
            urgency += 0.2
        
        # Decrease urgency for mean reversion
        if 'mean_reversion' in strategy.get('name', '').lower():
            urgency -= 0.2
        
        return max(0, min(1, urgency))
    
    async def _get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get current market data"""
        try:
            # Fetch ticker
            ticker = await self.exchange.fetch_ticker(symbol)
            
            # Fetch order book
            orderbook = await self.exchange.fetch_order_book(symbol, limit=10)
            
            # Calculate metrics
            best_bid = orderbook['bids'][0][0] if orderbook['bids'] else ticker['bid']
            best_ask = orderbook['asks'][0][0] if orderbook['asks'] else ticker['ask']
            
            return {
                'last_price': ticker['last'],
                'bid': best_bid,
                'ask': best_ask,
                'spread': best_ask - best_bid,
                'spread_pct': (best_ask - best_bid) / best_bid if best_bid > 0 else 0,
                'volume': ticker['baseVolume'],
                'avg_volume': ticker['baseVolume'],  # Simplified
                'volatility': ticker.get('percentage', 0) / 100  # Daily volatility proxy
            }
            
        except Exception as e:
            logger.error(f"Failed to get market data: {e}")
            # Return defaults
            return {
                'last_price': 0,
                'bid': 0,
                'ask': 0,
                'spread': 0,
                'spread_pct': 0,
                'volume': 0,
                'avg_volume': 0,
                'volatility': 0.02
            }
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        return f"BEAST_{datetime.now().timestamp()}_{np.random.randint(1000, 9999)}"
    
    async def start_monitoring(self):
        """Start order monitoring task"""
        if not self.monitoring_task:
            self.monitoring_task = asyncio.create_task(self._monitor_orders())
            logger.info("Order monitoring started")
    
    async def stop_monitoring(self):
        """Stop order monitoring"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            logger.info("Order monitoring stopped")
    
    async def _monitor_orders(self):
        """Monitor active orders and update status"""
        while True:
            try:
                # Check all active orders
                for order_id, order in list(self.active_orders.items()):
                    if order.status in [OrderStatus.SUBMITTED, OrderStatus.PARTIAL]:
                        # Update order status
                        try:
                            exchange_order = await self.exchange.fetch_order(
                                order.metadata.get('exchange_order_id'),
                                order.symbol
                            )
                            
                            # Update based on exchange status
                            if exchange_order['filled'] > order.filled_quantity:
                                fill_qty = exchange_order['filled'] - order.filled_quantity
                                order.update_fill(
                                    fill_qty,
                                    exchange_order['average'],
                                    exchange_order.get('fee', {}).get('cost', 0)
                                )
                                
                                # Emit fill event
                                await self.fill_stream.put({
                                    'order_id': order_id,
                                    'symbol': order.symbol,
                                    'fill_qty': fill_qty,
                                    'fill_price': exchange_order['average']
                                })
                                
                        except Exception as e:
                            logger.error(f"Error monitoring order {order_id}: {e}")
                
                # Check for exit triggers
                await self._check_exit_triggers()
                
                # Sleep before next check
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _check_exit_triggers(self):
        """Check if any exit orders should be triggered"""
        # This would check stop loss and take profit conditions
        # For positions and trigger orders accordingly
        pass
    
    def get_order_summary(self) -> Dict[str, Any]:
        """Get order execution summary"""
        active_value = sum(
            order.quantity * order.price for order in self.active_orders.values()
            if order.price
        )
        
        return {
            'active_orders': len(self.active_orders),
            'active_value': active_value,
            'execution_metrics': self.execution_metrics,
            'pending_orders': len([o for o in self.active_orders.values() if o.status == OrderStatus.PENDING]),
            'partial_fills': len([o for o in self.active_orders.values() if o.status == OrderStatus.PARTIAL])
        }
    
    def export_execution_report(self, filepath: str):
        """Export execution report"""
        report = {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'summary': self.get_order_summary(),
            'active_orders': [
                {
                    'id': order.id,
                    'symbol': order.symbol,
                    'side': order.side,
                    'type': order.type.value,
                    'quantity': order.quantity,
                    'filled': order.filled_quantity,
                    'avg_price': order.avg_fill_price,
                    'status': order.status.value,
                    'created_at': order.created_at.isoformat()
                }
                for order in self.active_orders.values()
            ],
            'recent_executions': [
                {
                    'id': order.id,
                    'symbol': order.symbol,
                    'side': order.side,
                    'quantity': order.filled_quantity,
                    'avg_price': order.avg_fill_price,
                    'slippage': order.slippage,
                    'fees': order.fees,
                    'execution_time': (order.updated_at - order.created_at).total_seconds()
                }
                for order in list(self.order_history)[-20:]
                if order.status == OrderStatus.FILLED
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Execution report exported to {filepath}")
    
    def is_healthy(self) -> bool:
        """Check if order manager is healthy"""
        # Check exchange connection
        if not self.exchange:
            return False
        
        # Check for stuck orders
        stuck_orders = [
            o for o in self.active_orders.values()
            if o.status == OrderStatus.SUBMITTED and
            (datetime.now(timezone.utc) - o.created_at).total_seconds() > 300
        ]
        
        if len(stuck_orders) > 5:
            return False
        
        return True