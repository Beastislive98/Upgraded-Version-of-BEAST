"""
BEAST Trading System - JSON Bridge
Handles Python to C++ communication for trade execution
"""

import json
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone
from decimal import Decimal
import asyncio
import aiofiles
from pathlib import Path
import hashlib
import hmac
from dataclasses import dataclass, asdict
from enum import Enum
import struct
import numpy as np

from config.settings import config
from utils.logger import get_logger

logger = get_logger(__name__)

class OrderType(Enum):
    """Order types supported by C++ execution engine"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"
    OCO = "OCO"  # One-Cancels-Other
    ICEBERG = "ICEBERG"

class OrderSide(Enum):
    """Order sides"""
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    """Order status"""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

@dataclass
class TradeSignal:
    """Standardized trade signal for C++ engine"""
    # Core fields
    signal_id: str
    timestamp: datetime
    symbol: str
    action: str  # TRADE or NO_TRADE
    side: OrderSide
    order_type: OrderType
    
    # Trade parameters
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    limit_price: Optional[Decimal] = None
    
    # Risk parameters
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    trailing_stop_distance: Optional[Decimal] = None
    max_slippage: Optional[Decimal] = None
    
    # Strategy information
    strategy_name: str = ""
    strategy_type: str = ""
    confidence: float = 0.0
    
    # Execution parameters
    time_in_force: str = "GTC"  # GTC, IOC, FOK, GTD
    expire_time: Optional[datetime] = None
    reduce_only: bool = False
    post_only: bool = False
    
    # Additional metadata
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization"""
        data = asdict(self)
        
        # Convert datetime objects
        data['timestamp'] = self.timestamp.isoformat()
        if self.expire_time:
            data['expire_time'] = self.expire_time.isoformat()
        
        # Convert Decimal to string for precision
        decimal_fields = ['quantity', 'price', 'stop_price', 'limit_price', 
                         'stop_loss', 'take_profit', 'trailing_stop_distance', 'max_slippage']
        for field in decimal_fields:
            if data.get(field) is not None:
                data[field] = str(data[field])
        
        # Convert enums
        data['side'] = self.side.value if isinstance(self.side, OrderSide) else self.side
        data['order_type'] = self.order_type.value if isinstance(self.order_type, OrderType) else self.order_type
        
        return data

@dataclass
class ExecutionResponse:
    """Response from C++ execution engine"""
    signal_id: str
    order_id: str
    status: OrderStatus
    timestamp: datetime
    executed_quantity: Decimal = Decimal('0')
    executed_price: Decimal = Decimal('0')
    commission: Decimal = Decimal('0')
    error_code: Optional[int] = None
    error_message: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionResponse':
        """Create from dictionary"""
        # Convert timestamp
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        # Convert Decimals
        decimal_fields = ['executed_quantity', 'executed_price', 'commission']
        for field in decimal_fields:
            if field in data and data[field] is not None:
                data[field] = Decimal(str(data[field]))
        
        # Convert status
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = OrderStatus(data['status'])
        
        return cls(**data)

class JSONBridge:
    """
    Bridge between Python analysis engine and C++ execution engine
    Handles serialization, validation, and communication
    """
    
    def __init__(self):
        self.config = config
        self.signal_queue_path = Path(config.exchange.get('signal_queue_path', './signals/pending/'))
        self.response_queue_path = Path(config.exchange.get('response_queue_path', './signals/responses/'))
        self.archive_path = Path(config.exchange.get('signal_archive_path', './signals/archive/'))
        
        # Create directories
        self.signal_queue_path.mkdir(parents=True, exist_ok=True)
        self.response_queue_path.mkdir(parents=True, exist_ok=True)
        self.archive_path.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.signal_count = 0
        self.response_count = 0
        self.error_count = 0
        
        # Signal tracking
        self.pending_signals = {}
        
        # Security
        self.signing_key = config.exchange.get('bridge_signing_key', 'default_key').encode()
        
        logger.info("JSONBridge initialized")
    
    async def send_signal(self, signal: TradeSignal) -> bool:
        """
        Send trading signal to C++ execution engine
        
        Returns:
            bool: True if signal was successfully queued
        """
        try:
            # Validate signal
            if not self._validate_signal(signal):
                logger.error(f"Invalid signal: {signal.signal_id}")
                return False
            
            # Prepare signal data
            signal_data = {
                'header': {
                    'version': '1.0',
                    'source': 'python_beast',
                    'target': 'cpp_executor'
                },
                'signal': signal.to_dict()
            }
            
            # Add signature for security
            signature = self._sign_data(signal_data)
            signal_data['signature'] = signature
            
            # Write to signal queue
            filename = f"{signal.signal_id}_{signal.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.signal_queue_path / filename
            
            async with aiofiles.open(filepath, 'w') as f:
                await f.write(json.dumps(signal_data, indent=2))
            
            # Track pending signal
            self.pending_signals[signal.signal_id] = {
                'signal': signal,
                'filepath': filepath,
                'timestamp': datetime.now(timezone.utc)
            }
            
            self.signal_count += 1
            logger.info(f"Signal queued: {signal.signal_id} for {signal.symbol}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending signal: {e}")
            self.error_count += 1
            return False
    
    async def check_responses(self) -> List[ExecutionResponse]:
        """
        Check for responses from C++ execution engine
        
        Returns:
            List of execution responses
        """
        responses = []
        
        try:
            # List response files
            response_files = sorted(self.response_queue_path.glob("*.json"))
            
            for filepath in response_files:
                try:
                    # Read response
                    async with aiofiles.open(filepath, 'r') as f:
                        content = await f.read()
                        data = json.loads(content)
                    
                    # Verify signature
                    if not self._verify_signature(data):
                        logger.warning(f"Invalid signature in response: {filepath}")
                        continue
                    
                    # Parse response
                    response = ExecutionResponse.from_dict(data['response'])
                    responses.append(response)
                    
                    # Update pending signals
                    if response.signal_id in self.pending_signals:
                        del self.pending_signals[response.signal_id]
                    
                    # Archive response
                    archive_file = self.archive_path / f"response_{filepath.name}"
                    filepath.rename(archive_file)
                    
                    self.response_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing response {filepath}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Error checking responses: {e}")
        
        return responses
    
    def create_trade_signal(
        self,
        symbol: str,
        strategy: Dict[str, Any],
        risk_params: Dict[str, Any],
        decision: Dict[str, Any]
    ) -> TradeSignal:
        """
        Create standardized trade signal from strategy recommendation
        """
        # Generate unique signal ID
        signal_id = self._generate_signal_id(symbol, strategy['name'])
        
        # Determine order side
        direction = decision.get('signals', {}).get('direction', 'long')
        side = OrderSide.BUY if direction == 'long' else OrderSide.SELL
        
        # Determine order type based on strategy
        order_type = self._determine_order_type(strategy)
        
        # Calculate quantities and prices
        quantity = self._calculate_quantity(
            symbol,
            strategy['position_size'],
            risk_params
        )
        
        # Get current price (should come from market data)
        current_price = Decimal(str(decision.get('market_context', {}).get('current_price', 0)))
        
        # Create signal
        signal = TradeSignal(
            signal_id=signal_id,
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            action="TRADE",
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=current_price if order_type != OrderType.MARKET else None,
            stop_loss=self._calculate_stop_loss(current_price, side, risk_params),
            take_profit=self._calculate_take_profit(current_price, side, risk_params),
            strategy_name=strategy['name'],
            strategy_type=strategy['type'],
            confidence=decision.get('confidence', 0.0),
            metadata={
                'decision_id': decision.get('decision_id'),
                'risk_score': risk_params.get('risk_score'),
                'expected_return': strategy.get('risk_reward_ratio', 2.0)
            }
        )
        
        return signal
    
    def create_close_signal(
        self,
        symbol: str,
        position: Dict[str, Any],
        reason: str
    ) -> TradeSignal:
        """
        Create signal to close existing position
        """
        # Generate close signal ID
        signal_id = self._generate_signal_id(symbol, f"close_{reason}")
        
        # Opposite side to close
        side = OrderSide.SELL if position['side'] == 'BUY' else OrderSide.BUY
        
        signal = TradeSignal(
            signal_id=signal_id,
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            action="TRADE",
            side=side,
            order_type=OrderType.MARKET,
            quantity=Decimal(str(position['quantity'])),
            reduce_only=True,
            strategy_name=f"close_{reason}",
            strategy_type="close",
            confidence=1.0,
            metadata={
                'original_position_id': position.get('position_id'),
                'close_reason': reason,
                'position_pnl': position.get('unrealized_pnl', 0)
            }
        )
        
        return signal
    
    def _validate_signal(self, signal: TradeSignal) -> bool:
        """Validate signal before sending"""
        # Check required fields
        if not signal.signal_id or not signal.symbol:
            return False
        
        # Validate quantity
        if signal.quantity <= 0:
            return False
        
        # Validate price for limit orders
        if signal.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            if not signal.price or signal.price <= 0:
                return False
        
        # Validate stop orders
        if signal.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            if not signal.stop_price or signal.stop_price <= 0:
                return False
        
        # Validate risk parameters
        if signal.stop_loss and signal.stop_loss <= 0:
            return False
        
        if signal.take_profit and signal.take_profit <= 0:
            return False
        
        return True
    
    def _generate_signal_id(self, symbol: str, strategy: str) -> str:
        """Generate unique signal ID"""
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')
        components = f"{symbol}_{strategy}_{timestamp}"
        return hashlib.md5(components.encode()).hexdigest()[:16]
    
    def _determine_order_type(self, strategy: Dict[str, Any]) -> OrderType:
        """Determine order type based on strategy"""
        strategy_type = strategy.get('type', '')
        
        if strategy_type == 'market_making':
            return OrderType.LIMIT
        elif strategy_type == 'scalping':
            return OrderType.MARKET
        elif strategy_type == 'arbitrage':
            return OrderType.LIMIT
        elif 'stop' in strategy.get('name', '').lower():
            return OrderType.STOP_LIMIT
        else:
            return OrderType.MARKET
    
    def _calculate_quantity(
        self,
        symbol: str,
        position_size_pct: float,
        risk_params: Dict[str, Any]
    ) -> Decimal:
        """Calculate order quantity based on position sizing"""
        # Get account balance (would come from capital manager)
        account_balance = Decimal(str(risk_params.get('account_balance', 100000)))
        
        # Calculate position value
        position_value = account_balance * Decimal(str(position_size_pct))
        
        # Get current price
        current_price = Decimal(str(risk_params.get('current_price', 1)))
        
        # Calculate quantity
        quantity = position_value / current_price
        
        # Round to appropriate decimals based on symbol
        # This should be symbol-specific in production
        if 'BTC' in symbol:
            quantity = quantity.quantize(Decimal('0.001'))
        else:
            quantity = quantity.quantize(Decimal('0.01'))
        
        return quantity
    
    def _calculate_stop_loss(
        self,
        price: Decimal,
        side: OrderSide,
        risk_params: Dict[str, Any]
    ) -> Decimal:
        """Calculate stop loss price"""
        stop_loss_pct = Decimal(str(risk_params.get('stop_loss', 0.02)))
        
        if side == OrderSide.BUY:
            stop_loss = price * (Decimal('1') - stop_loss_pct)
        else:
            stop_loss = price * (Decimal('1') + stop_loss_pct)
        
        return stop_loss.quantize(Decimal('0.01'))
    
    def _calculate_take_profit(
        self,
        price: Decimal,
        side: OrderSide,
        risk_params: Dict[str, Any]
    ) -> Decimal:
        """Calculate take profit price"""
        take_profit_pct = Decimal(str(risk_params.get('take_profit', 0.06)))
        
        if side == OrderSide.BUY:
            take_profit = price * (Decimal('1') + take_profit_pct)
        else:
            take_profit = price * (Decimal('1') - take_profit_pct)
        
        return take_profit.quantize(Decimal('0.01'))
    
    def _sign_data(self, data: Dict[str, Any]) -> str:
        """Sign data for security"""
        # Convert to canonical JSON string
        canonical = json.dumps(data, sort_keys=True, separators=(',', ':'))
        
        # Create HMAC signature
        signature = hmac.new(
            self.signing_key,
            canonical.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def _verify_signature(self, data: Dict[str, Any]) -> bool:
        """Verify data signature"""
        if 'signature' not in data:
            return False
        
        provided_signature = data.pop('signature')
        expected_signature = self._sign_data(data)
        
        return hmac.compare_digest(provided_signature, expected_signature)
    
    async def get_pending_signals(self) -> List[Dict[str, Any]]:
        """Get list of pending signals"""
        pending = []
        
        for signal_id, info in self.pending_signals.items():
            age = (datetime.now(timezone.utc) - info['timestamp']).total_seconds()
            pending.append({
                'signal_id': signal_id,
                'symbol': info['signal'].symbol,
                'side': info['signal'].side.value,
                'quantity': str(info['signal'].quantity),
                'age_seconds': age,
                'filepath': str(info['filepath'])
            })
        
        return pending
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get bridge statistics"""
        return {
            'signals_sent': self.signal_count,
            'responses_received': self.response_count,
            'errors': self.error_count,
            'pending_signals': len(self.pending_signals),
            'signal_queue_size': len(list(self.signal_queue_path.glob("*.json"))),
            'response_queue_size': len(list(self.response_queue_path.glob("*.json")))
        }
    
    async def cleanup_old_signals(self, max_age_hours: int = 24):
        """Clean up old signals and responses"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        
        # Clean pending signals
        to_remove = []
        for signal_id, info in self.pending_signals.items():
            if info['timestamp'] < cutoff_time:
                to_remove.append(signal_id)
        
        for signal_id in to_remove:
            logger.warning(f"Removing stale signal: {signal_id}")
            del self.pending_signals[signal_id]
        
        # Clean old files in archive
        for filepath in self.archive_path.glob("*.json"):
            try:
                stat = filepath.stat()
                file_time = datetime.fromtimestamp(stat.st_mtime, timezone.utc)
                if file_time < cutoff_time:
                    filepath.unlink()
                    logger.debug(f"Deleted old archive file: {filepath}")
            except Exception as e:
                logger.error(f"Error cleaning up {filepath}: {e}")
    
    def create_binary_signal(self, signal: TradeSignal) -> bytes:
        """
        Create binary format signal for ultra-low latency C++ communication
        Optional: for HFT strategies
        """
        # Binary format: [header(8)] [signal_id(16)] [timestamp(8)] [symbol(12)] ...
        # This is a simplified example
        
        format_string = '8s16sQ12sIffff'  # Header, ID, timestamp, symbol, side, quantities...
        
        header = b'BEAST001'  # Version header
        signal_id_bytes = signal.signal_id.encode()[:16].ljust(16, b'\0')
        timestamp = int(signal.timestamp.timestamp() * 1000000)  # Microseconds
        symbol_bytes = signal.symbol.encode()[:12].ljust(12, b'\0')
        side_int = 1 if signal.side == OrderSide.BUY else 2
        
        # Convert decimals to floats (loss of precision for speed)
        quantity_float = float(signal.quantity)
        price_float = float(signal.price) if signal.price else 0.0
        stop_loss_float = float(signal.stop_loss) if signal.stop_loss else 0.0
        take_profit_float = float(signal.take_profit) if signal.take_profit else 0.0
        
        binary_data = struct.pack(
            format_string,
            header,
            signal_id_bytes,
            timestamp,
            symbol_bytes,
            side_int,
            quantity_float,
            price_float,
            stop_loss_float,
            take_profit_float
        )
        
        return binary_data