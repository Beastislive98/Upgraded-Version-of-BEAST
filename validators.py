"""
BEAST Trading System - Validators
Comprehensive data validation for all system components
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timezone, timedelta
from decimal import Decimal, InvalidOperation
import pandas as pd
import numpy as np
import re
from dataclasses import dataclass

from config.settings import config

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of validation check"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]
    
    def add_error(self, error: str):
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        self.warnings.append(warning)

class Validators:
    """Static validation methods for all data types"""
    
    # Symbol validation patterns
    SYMBOL_PATTERNS = {
        'crypto_spot': re.compile(r'^[A-Z]{2,10}-[A-Z]{3,4}$'),  # BTC-USD
        'crypto_perp': re.compile(r'^[A-Z]{2,10}-[A-Z]{3,4}-PERP$'),  # BTC-USD-PERP
        'forex': re.compile(r'^[A-Z]{3}/[A-Z]{3}$'),  # EUR/USD
        'stock': re.compile(r'^[A-Z]{1,5}$')  # AAPL
    }
    
    # Price bounds by asset type
    PRICE_BOUNDS = {
        'BTC': (1000, 1000000),
        'ETH': (50, 50000),
        'DEFAULT': (0.00001, 1000000)
    }
    
    @staticmethod
    def validate_symbol(symbol: str) -> ValidationResult:
        """Validate trading symbol format"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], metadata={})
        
        if not symbol:
            result.add_error("Symbol is empty")
            return result
        
        # Check length
        if len(symbol) < 3 or len(symbol) > 20:
            result.add_error(f"Invalid symbol length: {len(symbol)}")
        
        # Check format
        valid_format = False
        for pattern_name, pattern in Validators.SYMBOL_PATTERNS.items():
            if pattern.match(symbol):
                valid_format = True
                result.metadata['symbol_type'] = pattern_name
                break
        
        if not valid_format:
            result.add_error(f"Invalid symbol format: {symbol}")
        
        # Check if in allowed list
        if hasattr(config.trading, 'enabled_pairs'):
            if symbol not in config.trading.enabled_pairs:
                result.add_warning(f"Symbol not in enabled pairs: {symbol}")
        
        return result
    
    @staticmethod
    def validate_price(price: Union[float, Decimal, str], symbol: str = None) -> ValidationResult:
        """Validate price value"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], metadata={})
        
        try:
            # Convert to Decimal for precision
            if isinstance(price, str):
                price_decimal = Decimal(price)
            elif isinstance(price, float):
                price_decimal = Decimal(str(price))
            else:
                price_decimal = price
            
            result.metadata['price_decimal'] = price_decimal
            
            # Check if positive
            if price_decimal <= 0:
                result.add_error(f"Price must be positive: {price_decimal}")
            
            # Check bounds based on symbol
            if symbol:
                base_asset = symbol.split('-')[0] if '-' in symbol else symbol
                min_price, max_price = Validators.PRICE_BOUNDS.get(
                    base_asset, 
                    Validators.PRICE_BOUNDS['DEFAULT']
                )
                
                if price_decimal < min_price:
                    result.add_error(f"Price below minimum for {base_asset}: {price_decimal} < {min_price}")
                elif price_decimal > max_price:
                    result.add_error(f"Price above maximum for {base_asset}: {price_decimal} > {max_price}")
            
            # Check decimal places
            decimal_places = abs(price_decimal.as_tuple().exponent)
            if decimal_places > 8:
                result.add_warning(f"Excessive decimal places: {decimal_places}")
            
        except (InvalidOperation, ValueError) as e:
            result.add_error(f"Invalid price format: {e}")
        
        return result
    
    @staticmethod
    def validate_quantity(
        quantity: Union[float, Decimal, str], 
        symbol: str = None,
        min_size: Optional[float] = None
    ) -> ValidationResult:
        """Validate order quantity"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], metadata={})
        
        try:
            # Convert to Decimal
            if isinstance(quantity, str):
                qty_decimal = Decimal(quantity)
            elif isinstance(quantity, float):
                qty_decimal = Decimal(str(quantity))
            else:
                qty_decimal = quantity
            
            result.metadata['quantity_decimal'] = qty_decimal
            
            # Check if positive
            if qty_decimal <= 0:
                result.add_error(f"Quantity must be positive: {qty_decimal}")
            
            # Check minimum size
            if min_size and qty_decimal < min_size:
                result.add_error(f"Quantity below minimum: {qty_decimal} < {min_size}")
            
            # Symbol-specific validation
            if symbol and 'BTC' in symbol:
                if qty_decimal < Decimal('0.0001'):
                    result.add_error(f"BTC quantity too small: {qty_decimal}")
                elif qty_decimal > Decimal('10000'):
                    result.add_warning(f"Unusually large BTC quantity: {qty_decimal}")
            
        except (InvalidOperation, ValueError) as e:
            result.add_error(f"Invalid quantity format: {e}")
        
        return result
    
    @staticmethod
    def validate_timestamp(
        timestamp: Union[datetime, str, int, float],
        max_age_seconds: Optional[int] = None
    ) -> ValidationResult:
        """Validate timestamp"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], metadata={})
        
        try:
            # Convert to datetime
            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            elif isinstance(timestamp, (int, float)):
                # Assume Unix timestamp
                if timestamp > 1e10:  # Milliseconds
                    dt = datetime.fromtimestamp(timestamp / 1000, timezone.utc)
                else:
                    dt = datetime.fromtimestamp(timestamp, timezone.utc)
            else:
                dt = timestamp
            
            # Ensure timezone aware
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            
            result.metadata['datetime'] = dt
            
            # Check if future
            now = datetime.now(timezone.utc)
            if dt > now:
                result.add_error(f"Timestamp is in the future: {dt}")
            
            # Check age
            age_seconds = (now - dt).total_seconds()
            result.metadata['age_seconds'] = age_seconds
            
            if max_age_seconds and age_seconds > max_age_seconds:
                result.add_error(f"Timestamp too old: {age_seconds}s > {max_age_seconds}s")
            
            # Reasonable bounds check (not before 2009 - Bitcoin genesis)
            if dt.year < 2009:
                result.add_error(f"Timestamp unreasonably old: {dt}")
            
        except Exception as e:
            result.add_error(f"Invalid timestamp format: {e}")
        
        return result
    
    @staticmethod
    def validate_percentage(
        value: Union[float, Decimal, str],
        min_val: float = 0.0,
        max_val: float = 1.0,
        name: str = "percentage"
    ) -> ValidationResult:
        """Validate percentage value"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], metadata={})
        
        try:
            # Convert to float
            if isinstance(value, str):
                float_val = float(value)
            elif isinstance(value, Decimal):
                float_val = float(value)
            else:
                float_val = value
            
            result.metadata['float_value'] = float_val
            
            # Check bounds
            if float_val < min_val:
                result.add_error(f"{name} below minimum: {float_val} < {min_val}")
            elif float_val > max_val:
                result.add_error(f"{name} above maximum: {float_val} > {max_val}")
            
            # Common percentage ranges
            if 0 <= float_val <= 1:
                result.metadata['format'] = 'decimal'
            elif 0 <= float_val <= 100:
                result.metadata['format'] = 'percentage'
                result.add_warning(f"Value appears to be in percentage format (0-100): {float_val}")
            
        except (ValueError, TypeError) as e:
            result.add_error(f"Invalid {name} format: {e}")
        
        return result

def validate_ohlcv_data(df: pd.DataFrame) -> ValidationResult:
    """Validate OHLCV dataframe"""
    result = ValidationResult(is_valid=True, errors=[], warnings=[], metadata={})
    
    if df is None or df.empty:
        result.add_error("OHLCV data is empty")
        return result
    
    # Check required columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        result.add_error(f"Missing required columns: {missing_columns}")
        return result
    
    # Check data types
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            result.add_error(f"Column '{col}' is not numeric")
    
    # Check OHLC relationships
    ohlc_valid = (
        (df['high'] >= df['low']).all() and
        (df['high'] >= df['open']).all() and
        (df['high'] >= df['close']).all() and
        (df['low'] <= df['open']).all() and
        (df['low'] <= df['close']).all()
    )
    
    if not ohlc_valid:
        result.add_error("Invalid OHLC relationships detected")
    
    # Check for negative prices
    if (df[['open', 'high', 'low', 'close']] < 0).any().any():
        result.add_error("Negative prices detected")
    
    # Check for zero prices
    if (df[['open', 'high', 'low', 'close']] == 0).any().any():
        result.add_warning("Zero prices detected")
    
    # Check for missing values
    missing_count = df[required_columns].isna().sum().sum()
    if missing_count > 0:
        missing_pct = missing_count / (len(df) * len(required_columns)) * 100
        if missing_pct > 5:
            result.add_error(f"Too many missing values: {missing_pct:.1f}%")
        else:
            result.add_warning(f"Missing values detected: {missing_count}")
    
    # Check index
    if not isinstance(df.index, pd.DatetimeIndex):
        result.add_warning("Index is not DatetimeIndex")
    else:
        # Check if sorted
        if not df.index.is_monotonic_increasing:
            result.add_error("Data is not sorted by time")
        
        # Check for duplicates
        if df.index.has_duplicates:
            result.add_error("Duplicate timestamps detected")
    
    # Metadata
    result.metadata['rows'] = len(df)
    result.metadata['time_range'] = {
        'start': df.index[0] if len(df) > 0 else None,
        'end': df.index[-1] if len(df) > 0 else None
    }
    
    return result

def validate_analysis_results(results: Dict[str, Any]) -> bool:
    """Quick validation of analysis results structure"""
    if not isinstance(results, dict):
        return False
    
    # Check for at least one successful analysis
    successful_modules = 0
    for module, data in results.items():
        if isinstance(data, dict) and data.get('status') != 'failed':
            successful_modules += 1
    
    return successful_modules >= 1

def validate_strategy_parameters(
    strategy_name: str,
    parameters: Dict[str, Any]
) -> ValidationResult:
    """Validate strategy-specific parameters"""
    result = ValidationResult(is_valid=True, errors=[], warnings=[], metadata={})
    
    # Common parameter validation
    if 'position_size' in parameters:
        pct_result = Validators.validate_percentage(
            parameters['position_size'],
            min_val=0.001,
            max_val=0.1,
            name="position_size"
        )
        if not pct_result.is_valid:
            result.errors.extend(pct_result.errors)
            result.is_valid = False
    
    if 'stop_loss' in parameters:
        sl_result = Validators.validate_percentage(
            parameters['stop_loss'],
            min_val=0.001,
            max_val=0.5,
            name="stop_loss"
        )
        if not sl_result.is_valid:
            result.errors.extend(sl_result.errors)
            result.is_valid = False
    
    # Strategy-specific validation
    if 'spread' in strategy_name:
        if 'spread_width' not in parameters:
            result.add_error("Spread strategy requires 'spread_width' parameter")
        elif parameters['spread_width'] <= 0:
            result.add_error("Spread width must be positive")
    
    if 'arbitrage' in strategy_name:
        if 'min_spread' not in parameters:
            result.add_error("Arbitrage strategy requires 'min_spread' parameter")
    
    return result

def validate_risk_parameters(risk_params: Dict[str, Any]) -> ValidationResult:
    """Validate risk management parameters"""
    result = ValidationResult(is_valid=True, errors=[], warnings=[], metadata={})
    
    # Required parameters
    required = ['risk_score', 'max_position_size']
    for param in required:
        if param not in risk_params:
            result.add_error(f"Missing required risk parameter: {param}")
    
    # Risk score validation
    if 'risk_score' in risk_params:
        score_result = Validators.validate_percentage(
            risk_params['risk_score'],
            min_val=0.0,
            max_val=1.0,
            name="risk_score"
        )
        if not score_result.is_valid:
            result.errors.extend(score_result.errors)
            result.is_valid = False
    
    # Position size validation
    if 'max_position_size' in risk_params:
        if risk_params['max_position_size'] <= 0:
            result.add_error("Max position size must be positive")
        elif risk_params['max_position_size'] > 0.1:
            result.add_warning("Max position size exceeds 10% of portfolio")
    
    # Stop loss validation
    if 'stop_loss' in risk_params:
        if risk_params['stop_loss'] <= 0:
            result.add_error("Stop loss must be positive")
        elif risk_params['stop_loss'] > 0.1:
            result.add_warning("Stop loss exceeds 10%")
    
    return result

def validate_order_parameters(order: Dict[str, Any]) -> ValidationResult:
    """Validate order parameters before execution"""
    result = ValidationResult(is_valid=True, errors=[], warnings=[], metadata={})
    
    # Required fields
    required_fields = ['symbol', 'side', 'quantity', 'order_type']
    for field in required_fields:
        if field not in order:
            result.add_error(f"Missing required order field: {field}")
    
    # Symbol validation
    if 'symbol' in order:
        symbol_result = Validators.validate_symbol(order['symbol'])
        if not symbol_result.is_valid:
            result.errors.extend(symbol_result.errors)
            result.is_valid = False
    
    # Side validation
    if 'side' in order:
        if order['side'] not in ['BUY', 'SELL', 'buy', 'sell']:
            result.add_error(f"Invalid order side: {order['side']}")
    
    # Quantity validation
    if 'quantity' in order:
        qty_result = Validators.validate_quantity(order['quantity'], order.get('symbol'))
        if not qty_result.is_valid:
            result.errors.extend(qty_result.errors)
            result.is_valid = False
    
    # Order type validation
    valid_order_types = ['MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT']
    if 'order_type' in order:
        if order['order_type'] not in valid_order_types:
            result.add_error(f"Invalid order type: {order['order_type']}")
    
    # Price validation for limit orders
    if order.get('order_type') in ['LIMIT', 'STOP_LIMIT']:
        if 'price' not in order:
            result.add_error("Limit order requires price")
        else:
            price_result = Validators.validate_price(order['price'], order.get('symbol'))
            if not price_result.is_valid:
                result.errors.extend(price_result.errors)
                result.is_valid = False
    
    return result

def validate_market_data(market_data: Dict[str, Any]) -> ValidationResult:
    """Validate market data structure"""
    result = ValidationResult(is_valid=True, errors=[], warnings=[], metadata={})
    
    # Check data freshness
    if 'timestamp' in market_data:
        ts_result = Validators.validate_timestamp(
            market_data['timestamp'],
            max_age_seconds=300  # 5 minutes
        )
        if not ts_result.is_valid:
            result.errors.extend(ts_result.errors)
            result.is_valid = False
    
    # Validate prices
    price_fields = ['bid', 'ask', 'last', 'close']
    for field in price_fields:
        if field in market_data:
            price_result = Validators.validate_price(market_data[field])
            if not price_result.is_valid:
                result.add_warning(f"Invalid {field} price")
    
    # Check bid-ask spread
    if 'bid' in market_data and 'ask' in market_data:
        try:
            bid = float(market_data['bid'])
            ask = float(market_data['ask'])
            if bid >= ask:
                result.add_error("Bid price >= Ask price")
            else:
                spread_pct = (ask - bid) / ask * 100
                if spread_pct > 5:
                    result.add_warning(f"Large bid-ask spread: {spread_pct:.2f}%")
        except (ValueError, TypeError):
            result.add_error("Invalid bid/ask prices")
    
    # Validate volume
    if 'volume' in market_data:
        try:
            volume = float(market_data['volume'])
            if volume < 0:
                result.add_error("Negative volume")
            elif volume == 0:
                result.add_warning("Zero volume")
        except (ValueError, TypeError):
            result.add_error("Invalid volume format")
    
    return result

def validate_blockchain_data(blockchain_data: Dict[str, Any]) -> ValidationResult:
    """Validate blockchain-specific data"""
    result = ValidationResult(is_valid=True, errors=[], warnings=[], metadata={})
    
    # Validate funding rate
    if 'funding_rate' in blockchain_data:
        rate = blockchain_data['funding_rate']
        if isinstance(rate, (int, float)):
            if abs(rate) > 0.1:  # 10%
                result.add_warning(f"Extreme funding rate: {rate}")
        else:
            result.add_error("Invalid funding rate format")
    
    # Validate open interest
    if 'open_interest' in blockchain_data:
        oi = blockchain_data['open_interest']
        if isinstance(oi, (int, float)):
            if oi < 0:
                result.add_error("Negative open interest")
        else:
            result.add_error("Invalid open interest format")
    
    # Validate whale metrics
    if 'whale_transactions' in blockchain_data:
        if not isinstance(blockchain_data['whale_transactions'], (list, int)):
            result.add_error("Invalid whale transactions format")
    
    return result

def validate_confidence_score(confidence: float) -> ValidationResult:
    """Validate confidence score"""
    return Validators.validate_percentage(
        confidence,
        min_val=0.0,
        max_val=1.0,
        name="confidence"
    )

def validate_dataframe_index(df: pd.DataFrame) -> ValidationResult:
    """Validate dataframe index properties"""
    result = ValidationResult(is_valid=True, errors=[], warnings=[], metadata={})
    
    if not isinstance(df.index, pd.DatetimeIndex):
        result.add_error("Index is not DatetimeIndex")
        return result
    
    # Check timezone
    if df.index.tz is None:
        result.add_warning("Index is not timezone-aware")
    
    # Check monotonic
    if not df.index.is_monotonic_increasing:
        result.add_error("Index is not monotonic increasing")
    
    # Check duplicates
    if df.index.has_duplicates:
        dup_count = df.index.duplicated().sum()
        result.add_error(f"Index has {dup_count} duplicates")
    
    # Check frequency
    if len(df) > 1:
        time_diffs = pd.Series(df.index).diff().dropna()
        
        # Check for regular frequency
        if time_diffs.std() / time_diffs.mean() < 0.01:
            result.metadata['frequency'] = time_diffs.mode()[0]
            result.metadata['is_regular'] = True
        else:
            result.metadata['is_regular'] = False
            
            # Check for gaps
            median_diff = time_diffs.median()
            large_gaps = time_diffs[time_diffs > median_diff * 5]
            if len(large_gaps) > 0:
                result.add_warning(f"Found {len(large_gaps)} large time gaps")
    
    return result

# Convenience functions for common validations
def is_valid_symbol(symbol: str) -> bool:
    """Quick check if symbol is valid"""
    return Validators.validate_symbol(symbol).is_valid

def is_valid_price(price: Union[float, Decimal, str]) -> bool:
    """Quick check if price is valid"""
    return Validators.validate_price(price).is_valid

def is_valid_quantity(quantity: Union[float, Decimal, str]) -> bool:
    """Quick check if quantity is valid"""
    return Validators.validate_quantity(quantity).is_valid

def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize dataframe by removing invalid data"""
    if df.empty:
        return df
    
    # Remove rows with invalid OHLC relationships
    if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        valid_ohlc = (
            (df['high'] >= df['low']) &
            (df['high'] >= df['open']) &
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) &
            (df['low'] <= df['close'])
        )
        df = df[valid_ohlc]
    
    # Remove rows with negative or zero prices
    price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in df.columns]
    if price_cols:
        df = df[(df[price_cols] > 0).all(axis=1)]
    
    # Remove duplicate indices
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep='first')]
    
    # Sort by index
    df = df.sort_index()
    
    return df