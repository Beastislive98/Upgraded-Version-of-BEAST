"""
BEAST Trading System - Data Manager
Unified data collection, processing, and quality validation
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
import ccxt.async_support as ccxt
from collections import deque
import aiohttp
import json

from config.settings import config
from utils.logger import get_logger
from utils.cache import DataCache

logger = get_logger(__name__)

@dataclass
class DataQuality:
    """Data quality metrics"""
    completeness: float
    consistency: float
    recency: float
    outlier_ratio: float
    overall_score: float
    issues: List[str]
    
    def is_acceptable(self) -> bool:
        return self.overall_score >= 0.6 and self.completeness >= 0.9

class DataManager:
    """
    Consolidated data collection and processing
    Ensures high-quality data for trading decisions
    """
    
    def __init__(self, config):
        self.config = config
        self.exchange = None
        self.cache = DataCache()
        
        # Data buffers for streaming
        self.price_buffers = {}
        self.orderbook_buffers = {}
        
        # Initialize exchange connection
        self._init_exchange()
        
        # Performance metrics
        self.fetch_times = deque(maxlen=100)
        self.quality_scores = deque(maxlen=100)
        
        logger.info("DataManager initialized")
    
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
                
            logger.info(f"Connected to {self.config.exchange.exchange_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            raise
    
    async def collect_and_process(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Collect and process all data for a symbol
        Returns processed data with quality metrics
        """
        start_time = datetime.now()
        
        try:
            # Check cache first
            cached_data = self.cache.get(f"processed_{symbol}")
            if cached_data and self._is_data_fresh(cached_data):
                logger.debug(f"Using cached data for {symbol}")
                return cached_data
            
            # Collect raw data in parallel
            raw_data = await self._collect_raw_data(symbol)
            if not raw_data:
                logger.warning(f"Failed to collect data for {symbol}")
                return None
            
            # Process and validate data
            processed_data = self._process_data(raw_data)
            
            # Calculate quality metrics
            quality = self._calculate_quality(processed_data)
            processed_data['quality_metrics'] = quality.__dict__
            processed_data['quality_score'] = quality.overall_score
            
            # Only return if quality is acceptable
            if not quality.is_acceptable():
                logger.warning(f"Data quality below threshold for {symbol}: {quality.overall_score:.2f}")
                logger.warning(f"Issues: {quality.issues}")
                return None
            
            # Cache processed data
            self.cache.set(f"processed_{symbol}", processed_data, ttl=60)
            
            # Record metrics
            elapsed = (datetime.now() - start_time).total_seconds()
            self.fetch_times.append(elapsed)
            self.quality_scores.append(quality.overall_score)
            
            logger.debug(f"Data collection for {symbol} completed in {elapsed:.2f}s")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error collecting data for {symbol}: {e}")
            return None
    
    async def _collect_raw_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Collect all raw data sources in parallel"""
        tasks = {
            'ohlcv': self._fetch_ohlcv(symbol),
            'orderbook': self._fetch_orderbook(symbol),
            'trades': self._fetch_recent_trades(symbol),
            'ticker': self._fetch_ticker(symbol),
            'funding': self._fetch_funding_rate(symbol),
            'market_data': self._fetch_market_data(symbol)
        }
        
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        raw_data = {}
        for (key, _), result in zip(tasks.items(), results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to fetch {key} for {symbol}: {result}")
                if key == 'ohlcv':  # OHLCV is mandatory
                    return None
            else:
                raw_data[key] = result
        
        return raw_data
    
    async def _fetch_ohlcv(self, symbol: str) -> pd.DataFrame:
        """Fetch OHLCV data"""
        try:
            # Fetch different timeframes
            timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
            ohlcv_data = {}
            
            for tf in timeframes:
                if self.exchange.has['fetchOHLCV']:
                    data = await self.exchange.fetch_ohlcv(
                        symbol, 
                        timeframe=tf,
                        limit=self.config.trading.min_data_points
                    )
                    
                    df = pd.DataFrame(
                        data, 
                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    )
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    ohlcv_data[tf] = df
            
            # Return primary timeframe (1h) with additional timeframes attached
            primary_df = ohlcv_data.get('1h', ohlcv_data.get('15m'))
            primary_df.attrs['timeframes'] = ohlcv_data
            
            return primary_df
            
        except Exception as e:
            logger.error(f"OHLCV fetch error: {e}")
            raise
    
    async def _fetch_orderbook(self, symbol: str) -> Dict[str, Any]:
        """Fetch order book data"""
        try:
            if not self.exchange.has['fetchOrderBook']:
                return {}
                
            orderbook = await self.exchange.fetch_order_book(symbol, limit=50)
            
            # Calculate metrics
            bid_volume = sum(bid[1] for bid in orderbook['bids'][:10])
            ask_volume = sum(ask[1] for ask in orderbook['asks'][:10])
            
            spread = orderbook['asks'][0][0] - orderbook['bids'][0][0] if orderbook['asks'] and orderbook['bids'] else 0
            mid_price = (orderbook['asks'][0][0] + orderbook['bids'][0][0]) / 2 if orderbook['asks'] and orderbook['bids'] else 0
            
            return {
                'bids': orderbook['bids'][:20],
                'asks': orderbook['asks'][:20],
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'spread': spread,
                'spread_pct': (spread / mid_price * 100) if mid_price > 0 else 0,
                'order_flow_imbalance': (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0,
                'timestamp': datetime.now(timezone.utc)
            }
            
        except Exception as e:
            logger.warning(f"Orderbook fetch error: {e}")
            return {}
    
    async def _fetch_recent_trades(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch recent trades"""
        try:
            if not self.exchange.has['fetchTrades']:
                return []
                
            trades = await self.exchange.fetch_trades(symbol, limit=100)
            
            # Process trades for analysis
            processed_trades = []
            for trade in trades[-50:]:  # Last 50 trades
                processed_trades.append({
                    'timestamp': trade['timestamp'],
                    'price': trade['price'],
                    'amount': trade['amount'],
                    'side': trade['side'],
                    'cost': trade['cost']
                })
            
            return processed_trades
            
        except Exception as e:
            logger.warning(f"Trades fetch error: {e}")
            return []
    
    async def _fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """Fetch ticker data"""
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            
            return {
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'last': ticker['last'],
                'volume': ticker['baseVolume'],
                'quote_volume': ticker['quoteVolume'],
                'change': ticker['change'],
                'percentage': ticker['percentage'],
                'vwap': ticker.get('vwap', ticker['last']),
                'timestamp': ticker['timestamp']
            }
            
        except Exception as e:
            logger.warning(f"Ticker fetch error: {e}")
            return {}
    
    async def _fetch_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """Fetch funding rate for perpetual contracts"""
        try:
            # Only for perpetual futures
            if 'PERP' not in symbol and 'perpetual' not in self.exchange.options.get('defaultType', ''):
                return {}
            
            if hasattr(self.exchange, 'fetch_funding_rate'):
                funding = await self.exchange.fetch_funding_rate(symbol)
                return {
                    'rate': funding['fundingRate'],
                    'timestamp': funding['timestamp'],
                    'next_funding': funding.get('nextFundingTime')
                }
            
            return {}
            
        except Exception as e:
            logger.debug(f"Funding rate not available: {e}")
            return {}
    
    async def _fetch_market_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch additional market data (open interest, etc.)"""
        try:
            market_data = {}
            
            # Fetch open interest if available
            if hasattr(self.exchange, 'fetch_open_interest'):
                oi = await self.exchange.fetch_open_interest(symbol)
                market_data['open_interest'] = oi['openInterest']
                market_data['oi_timestamp'] = oi['timestamp']
            
            # Add more market data sources as needed
            
            return market_data
            
        except Exception as e:
            logger.debug(f"Market data fetch error: {e}")
            return {}
    
    def _process_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw data into analysis-ready format"""
        processed = {
            'symbol': raw_data.get('ticker', {}).get('symbol', 'UNKNOWN'),
            'timestamp': datetime.now(timezone.utc),
            'price_data': None,
            'market_data': {},
            'orderbook': {},
            'indicators': {}
        }
        
        # Process OHLCV data
        if 'ohlcv' in raw_data and raw_data['ohlcv'] is not None:
            df = raw_data['ohlcv']
            
            # Clean data (gentle approach)
            df = self._clean_ohlcv_data(df)
            
            # Add basic indicators
            df = self._add_basic_indicators(df)
            
            processed['price_data'] = df
            processed['indicators'] = self._extract_latest_indicators(df)
        
        # Process market data
        if 'ticker' in raw_data:
            processed['market_data'].update(raw_data['ticker'])
        
        if 'funding' in raw_data and raw_data['funding']:
            processed['market_data']['funding_rate'] = raw_data['funding'].get('rate', 0)
        
        if 'market_data' in raw_data:
            processed['market_data'].update(raw_data['market_data'])
        
        # Process orderbook
        if 'orderbook' in raw_data:
            processed['orderbook'] = raw_data['orderbook']
        
        # Process recent trades for whale detection
        if 'trades' in raw_data:
            processed['recent_trades'] = raw_data['trades']
            processed['whale_trades'] = self._detect_whale_trades(raw_data['trades'])
        
        return processed
    
    def _clean_ohlcv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean OHLCV data gently to preserve integrity"""
        # Remove exact duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Sort by timestamp
        df = df.sort_index()
        
        # Handle missing values (forward fill with limit)
        df = df.fillna(method='ffill', limit=2)
        
        # Remove rows with any remaining NaN in critical columns
        critical_cols = ['open', 'high', 'low', 'close', 'volume']
        df = df.dropna(subset=critical_cols)
        
        # Flag potential outliers but don't remove
        df['is_outlier'] = self._flag_outliers(df)
        
        return df
    
    def _flag_outliers(self, df: pd.DataFrame) -> pd.Series:
        """Flag outliers without removing them"""
        # Use rolling statistics for outlier detection
        window = 20
        
        # Calculate rolling mean and std
        rolling_mean = df['close'].rolling(window=window, center=True).mean()
        rolling_std = df['close'].rolling(window=window, center=True).std()
        
        # Flag values beyond 4 standard deviations
        upper_bound = rolling_mean + (4 * rolling_std)
        lower_bound = rolling_mean - (4 * rolling_std)
        
        is_outlier = (df['close'] > upper_bound) | (df['close'] < lower_bound)
        
        return is_outlier.fillna(False)
    
    def _add_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators needed by analysis modules"""
        # Simple Moving Averages
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price changes
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # High-Low spread
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        
        # ATR (Average True Range)
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.abs(df['high'] - df['close'].shift(1)),
            np.abs(df['low'] - df['close'].shift(1))
        )
        df['ATR_14'] = df['tr'].rolling(window=14).mean()
        
        return df
    
    def _extract_latest_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract latest indicator values"""
        latest = {}
        
        # Get the last valid row
        last_idx = df.index[-1]
        
        # Price data
        latest['close'] = df.loc[last_idx, 'close']
        latest['volume'] = df.loc[last_idx, 'volume']
        latest['returns'] = df.loc[last_idx, 'returns']
        
        # Moving averages
        for col in ['SMA_20', 'SMA_50', 'EMA_12', 'EMA_26']:
            if col in df.columns:
                latest[col] = df.loc[last_idx, col]
        
        # Volatility
        latest['volatility'] = df['returns'].rolling(window=20).std().iloc[-1]
        latest['ATR_14'] = df.loc[last_idx, 'ATR_14'] if 'ATR_14' in df.columns else None
        
        return latest
    
    def _detect_whale_trades(self, trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect potential whale trades"""
        if not trades:
            return []
        
        # Calculate trade sizes
        trade_sizes = [t['cost'] for t in trades]
        if not trade_sizes:
            return []
        
        # Use percentile-based detection
        p95 = np.percentile(trade_sizes, 95)
        p99 = np.percentile(trade_sizes, 99)
        
        whale_trades = []
        for trade in trades:
            if trade['cost'] >= p99:
                trade['whale_level'] = 'large'
                whale_trades.append(trade)
            elif trade['cost'] >= p95:
                trade['whale_level'] = 'medium'
                whale_trades.append(trade)
        
        return whale_trades
    
    def _calculate_quality(self, processed_data: Dict[str, Any]) -> DataQuality:
        """Calculate comprehensive data quality metrics"""
        issues = []
        
        # Check data completeness
        completeness = self._calculate_completeness(processed_data)
        if completeness < 0.9:
            issues.append(f"Low completeness: {completeness:.2%}")
        
        # Check data consistency
        consistency = self._calculate_consistency(processed_data)
        if consistency < 0.8:
            issues.append(f"Low consistency: {consistency:.2%}")
        
        # Check data recency
        recency = self._calculate_recency(processed_data)
        if recency < 0.8:
            issues.append("Stale data detected")
        
        # Check outlier ratio
        outlier_ratio = self._calculate_outlier_ratio(processed_data)
        if outlier_ratio > 0.05:
            issues.append(f"High outlier ratio: {outlier_ratio:.2%}")
        
        # Calculate overall score
        overall_score = (
            completeness * 0.4 +
            consistency * 0.3 +
            recency * 0.2 +
            (1 - outlier_ratio) * 0.1
        )
        
        return DataQuality(
            completeness=completeness,
            consistency=consistency,
            recency=recency,
            outlier_ratio=outlier_ratio,
            overall_score=overall_score,
            issues=issues
        )
    
    def _calculate_completeness(self, data: Dict[str, Any]) -> float:
        """Calculate data completeness score"""
        if 'price_data' not in data or data['price_data'] is None:
            return 0.0
        
        df = data['price_data']
        
        # Check for missing values
        total_expected = len(df) * len(['open', 'high', 'low', 'close', 'volume'])
        missing = df[['open', 'high', 'low', 'close', 'volume']].isna().sum().sum()
        
        completeness = 1 - (missing / total_expected)
        
        # Check if we have minimum required data points
        if len(df) < self.config.trading.min_data_points:
            completeness *= 0.5
        
        return completeness
    
    def _calculate_consistency(self, data: Dict[str, Any]) -> float:
        """Calculate data consistency score"""
        if 'price_data' not in data or data['price_data'] is None:
            return 0.0
        
        df = data['price_data']
        consistency_checks = []
        
        # Check OHLC relationships
        ohlc_valid = (
            (df['high'] >= df['low']).all() and
            (df['high'] >= df['open']).all() and
            (df['high'] >= df['close']).all() and
            (df['low'] <= df['open']).all() and
            (df['low'] <= df['close']).all()
        )
        consistency_checks.append(float(ohlc_valid))
        
        # Check for zero or negative prices
        positive_prices = (df[['open', 'high', 'low', 'close']] > 0).all().all()
        consistency_checks.append(float(positive_prices))
        
        # Check for reasonable price changes (less than 50% in one candle)
        price_changes = df['close'].pct_change().abs()
        reasonable_changes = (price_changes < 0.5).mean()
        consistency_checks.append(reasonable_changes)
        
        return np.mean(consistency_checks)
    
    def _calculate_recency(self, data: Dict[str, Any]) -> float:
        """Calculate data recency score"""
        if 'price_data' not in data or data['price_data'] is None:
            return 0.0
        
        df = data['price_data']
        
        # Check age of latest data
        latest_timestamp = df.index[-1]
        current_time = datetime.now(timezone.utc)
        
        # Convert to timezone-aware if needed
        if latest_timestamp.tzinfo is None:
            latest_timestamp = latest_timestamp.replace(tzinfo=timezone.utc)
        
        age_minutes = (current_time - latest_timestamp).total_seconds() / 60
        
        # Score based on age
        if age_minutes <= 1:
            return 1.0
        elif age_minutes <= 5:
            return 0.9
        elif age_minutes <= 15:
            return 0.7
        elif age_minutes <= 30:
            return 0.5
        else:
            return 0.0
    
    def _calculate_outlier_ratio(self, data: Dict[str, Any]) -> float:
        """Calculate outlier ratio"""
        if 'price_data' not in data or data['price_data'] is None:
            return 0.0
        
        df = data['price_data']
        
        if 'is_outlier' in df.columns:
            return df['is_outlier'].mean()
        
        return 0.0
    
    def _is_data_fresh(self, data: Dict[str, Any]) -> bool:
        """Check if cached data is still fresh"""
        if 'timestamp' not in data:
            return False
        
        data_age = (datetime.now(timezone.utc) - data['timestamp']).total_seconds()
        return data_age < self.config.trading.max_data_staleness_minutes * 60
    
    def is_healthy(self) -> bool:
        """Check if data manager is healthy"""
        try:
            # Check exchange connection
            if self.exchange is None:
                return False
            
            # Check recent fetch times
            if self.fetch_times:
                avg_fetch_time = np.mean(self.fetch_times)
                if avg_fetch_time > 10:  # More than 10 seconds average
                    return False
            
            # Check recent quality scores
            if self.quality_scores:
                avg_quality = np.mean(self.quality_scores)
                if avg_quality < 0.7:
                    return False
            
            return True
            
        except Exception:
            return False
    
    async def close(self):
        """Close connections and cleanup"""
        if self.exchange:
            await self.exchange.close()
        self.cache.clear()
        logger.info("DataManager closed")