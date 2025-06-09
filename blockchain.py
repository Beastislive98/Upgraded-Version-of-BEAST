"""
BEAST Trading System - Blockchain Analyzer
Comprehensive blockchain and crypto-specific analysis
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone, timedelta
import logging
from collections import defaultdict

from config.settings import config
from utils.logger import get_logger

logger = get_logger(__name__)

class BlockchainAnalyzer:
    """
    Analyzes blockchain-specific metrics for crypto trading
    Provides on-chain insights unavailable in traditional markets
    """
    
    def __init__(self, config):
        self.config = config
        self.blockchain_config = config.blockchain
        
        # API endpoints
        self.api_endpoints = {
            'coingecko': 'https://api.coingecko.com/api/v3',
            'glassnode': 'https://api.glassnode.com/v1',
            'cryptoquant': 'https://api.cryptoquant.com/v1'
        }
        
        # Cache for API data
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Metric thresholds
        self.thresholds = {
            'nvt_oversold': self.blockchain_config.nvt_oversold_threshold,
            'nvt_overbought': self.blockchain_config.nvt_overbought_threshold,
            'exchange_flow_significant': self.blockchain_config.exchange_flow_significance,
            'whale_btc': self.blockchain_config.whale_threshold_btc,
            'whale_eth': self.blockchain_config.whale_threshold_eth
        }
        
        logger.info("BlockchainAnalyzer initialized")
    
    async def analyze(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive blockchain analysis
        """
        result = {
            'status': 'analyzed',
            'metrics': {},
            'signals': {},
            'confidence': 0.0,
            'direction': None,
            'warnings': []
        }
        
        try:
            # Extract base asset from symbol
            base_asset = self._extract_base_asset(symbol)
            
            # Get price data
            price_data = data.get('price_data')
            if price_data is None:
                result['status'] = 'no_price_data'
                return result
            
            # Collect blockchain metrics
            metrics = {}
            
            # 1. On-chain metrics
            onchain = await self._analyze_onchain_metrics(base_asset, price_data)
            metrics['onchain'] = onchain
            
            # 2. Exchange flow analysis
            exchange_flows = await self._analyze_exchange_flows(base_asset)
            metrics['exchange_flows'] = exchange_flows
            
            # 3. Market structure
            market_structure = self._analyze_market_structure(data)
            metrics['market_structure'] = market_structure
            
            # 4. DeFi metrics (if applicable)
            if self._is_defi_asset(base_asset):
                defi_metrics = await self._analyze_defi_metrics(base_asset)
                metrics['defi'] = defi_metrics
            
            # 5. Whale activity
            whale_activity = self._analyze_whale_activity(base_asset, data)
            metrics['whale_activity'] = whale_activity
            
            # 6. Network health
            network_health = await self._analyze_network_health(base_asset)
            metrics['network_health'] = network_health
            
            result['metrics'] = metrics
            
            # Generate signals from metrics
            signals = self._generate_blockchain_signals(metrics)
            result['signals'] = signals
            
            # Calculate confidence and direction
            confidence, direction = self._calculate_confidence_and_direction(signals, metrics)
            result['confidence'] = confidence
            result['direction'] = direction
            
            # Add any warnings
            result['warnings'] = self._check_for_warnings(metrics)
            
        except Exception as e:
            logger.error(f"Blockchain analysis error for {symbol}: {e}")
            result['status'] = 'error'
            result['error'] = str(e)
        
        return result
    
    def _extract_base_asset(self, symbol: str) -> str:
        """Extract base asset from trading pair"""
        # Handle common formats: BTC-USD, BTCUSDT, BTC/USD
        for separator in ['-', '/', '']:
            if separator in symbol or separator == '':
                parts = symbol.split(separator) if separator else [symbol[:3], symbol[3:]]
                if len(parts) >= 2:
                    return parts[0].lower()
        return symbol.lower()
    
    async def _analyze_onchain_metrics(self, asset: str, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze on-chain metrics"""
        metrics = {
            'nvt_ratio': 0.0,
            'nvt_signal': None,
            'active_addresses': 0,
            'transaction_volume': 0.0,
            'realized_cap': 0.0,
            'mvrv_ratio': 0.0,
            'supply_in_profit': 0.0,
            'signals': []
        }
        
        try:
            # Get latest price
            latest_price = price_data['close'].iloc[-1]
            
            # Fetch on-chain data (simulated for now - replace with actual API calls)
            onchain_data = await self._fetch_onchain_data(asset)
            
            if onchain_data:
                # NVT Ratio calculation
                market_cap = latest_price * onchain_data.get('circulating_supply', 0)
                daily_tx_volume = onchain_data.get('transaction_volume_usd', 0)
                
                if daily_tx_volume > 0:
                    nvt_ratio = market_cap / (daily_tx_volume * 365)  # Annualized
                    metrics['nvt_ratio'] = nvt_ratio
                    
                    # Generate NVT signals
                    if nvt_ratio < self.thresholds['nvt_oversold']:
                        metrics['signals'].append(('nvt_undervalued', 0.8))
                        metrics['nvt_signal'] = 'undervalued'
                    elif nvt_ratio > self.thresholds['nvt_overbought']:
                        metrics['signals'].append(('nvt_overvalued', 0.8))
                        metrics['nvt_signal'] = 'overvalued'
                    else:
                        metrics['nvt_signal'] = 'fair_value'
                
                # Active addresses
                metrics['active_addresses'] = onchain_data.get('active_addresses', 0)
                
                # Network growth signals
                addr_growth = onchain_data.get('address_growth_rate', 0)
                if addr_growth > 0.05:  # 5% growth
                    metrics['signals'].append(('network_growth', 0.7))
                elif addr_growth < -0.05:  # 5% decline
                    metrics['signals'].append(('network_decline', 0.6))
                
                # MVRV Ratio (Market Value to Realized Value)
                realized_cap = onchain_data.get('realized_cap', 0)
                if realized_cap > 0:
                    mvrv = market_cap / realized_cap
                    metrics['mvrv_ratio'] = mvrv
                    
                    if mvrv < 1:
                        metrics['signals'].append(('mvrv_undervalued', 0.7))
                    elif mvrv > 3:
                        metrics['signals'].append(('mvrv_overvalued', 0.7))
                
                # Supply in profit
                profitable_supply = onchain_data.get('supply_in_profit_pct', 0.5)
                metrics['supply_in_profit'] = profitable_supply
                
                if profitable_supply > 0.9:
                    metrics['signals'].append(('high_profit_supply', 0.6))
                elif profitable_supply < 0.3:
                    metrics['signals'].append(('low_profit_supply', 0.6))
                    
        except Exception as e:
            logger.warning(f"On-chain metrics error: {e}")
        
        return metrics
    
    async def _analyze_exchange_flows(self, asset: str) -> Dict[str, Any]:
        """Analyze exchange inflows and outflows"""
        flows = {
            'net_flow': 0.0,
            'inflow_volume': 0.0,
            'outflow_volume': 0.0,
            'exchange_reserves': 0.0,
            'reserve_change': 0.0,
            'signals': []
        }
        
        try:
            # Fetch exchange flow data
            flow_data = await self._fetch_exchange_flow_data(asset)
            
            if flow_data:
                inflows = flow_data.get('inflow_volume', 0)
                outflows = flow_data.get('outflow_volume', 0)
                net_flow = outflows - inflows  # Positive = net outflow (bullish)
                
                flows['inflow_volume'] = inflows
                flows['outflow_volume'] = outflows
                flows['net_flow'] = net_flow
                
                # Exchange reserves
                reserves = flow_data.get('exchange_reserves', 0)
                reserve_change = flow_data.get('reserve_change_pct', 0)
                
                flows['exchange_reserves'] = reserves
                flows['reserve_change'] = reserve_change
                
                # Generate signals
                if abs(net_flow) > self.thresholds['exchange_flow_significant']:
                    if net_flow > 0:
                        flows['signals'].append(('exchange_outflow', 0.8))
                    else:
                        flows['signals'].append(('exchange_inflow', 0.8))
                
                if reserve_change < -0.05:  # 5% decrease
                    flows['signals'].append(('decreasing_reserves', 0.7))
                elif reserve_change > 0.05:  # 5% increase
                    flows['signals'].append(('increasing_reserves', 0.6))
                    
        except Exception as e:
            logger.warning(f"Exchange flow analysis error: {e}")
        
        return flows
    
    def _analyze_market_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze crypto market structure"""
        structure = {
            'funding_rate': 0.0,
            'open_interest': 0.0,
            'oi_change': 0.0,
            'basis': 0.0,
            'signals': []
        }
        
        try:
            market_data = data.get('market_data', {})
            
            # Funding rate
            funding_rate = market_data.get('funding_rate', 0)
            structure['funding_rate'] = funding_rate
            
            if funding_rate > 0.05:  # 5% (annualized)
                structure['signals'].append(('high_funding', 0.7))
            elif funding_rate < -0.03:  # -3%
                structure['signals'].append(('negative_funding', 0.6))
            
            # Open interest
            oi = market_data.get('open_interest', 0)
            oi_change = market_data.get('oi_change_pct', 0)
            
            structure['open_interest'] = oi
            structure['oi_change'] = oi_change
            
            if oi_change > 0.1:  # 10% increase
                structure['signals'].append(('increasing_oi', 0.6))
            elif oi_change < -0.1:  # 10% decrease
                structure['signals'].append(('decreasing_oi', 0.5))
            
            # Spot-futures basis
            spot_price = data.get('price_data', pd.DataFrame()).get('close', pd.Series()).iloc[-1] if 'price_data' in data else 0
            futures_price = market_data.get('futures_price', spot_price)
            
            if spot_price > 0:
                basis = (futures_price - spot_price) / spot_price
                structure['basis'] = basis
                
                if abs(basis) > 0.01:  # 1% basis
                    structure['signals'].append(('arbitrage_opportunity', 0.8))
                    
        except Exception as e:
            logger.warning(f"Market structure analysis error: {e}")
        
        return structure
    
    async def _analyze_defi_metrics(self, asset: str) -> Dict[str, Any]:
        """Analyze DeFi-specific metrics"""
        defi = {
            'tvl': 0.0,
            'tvl_change': 0.0,
            'protocol_revenue': 0.0,
            'utilization_rate': 0.0,
            'lending_rates': {},
            'signals': []
        }
        
        try:
            # Fetch DeFi data
            defi_data = await self._fetch_defi_data(asset)
            
            if defi_data:
                # Total Value Locked
                tvl = defi_data.get('tvl', 0)
                tvl_change = defi_data.get('tvl_change_24h', 0)
                
                defi['tvl'] = tvl
                defi['tvl_change'] = tvl_change
                
                if tvl_change > 0.1:  # 10% increase
                    defi['signals'].append(('increasing_tvl', 0.7))
                elif tvl_change < -0.1:  # 10% decrease
                    defi['signals'].append(('decreasing_tvl', 0.6))
                
                # Protocol metrics
                defi['protocol_revenue'] = defi_data.get('revenue_24h', 0)
                defi['utilization_rate'] = defi_data.get('utilization', 0)
                
                if defi['utilization_rate'] > 0.8:  # 80% utilization
                    defi['signals'].append(('high_utilization', 0.6))
                
                # Lending rates
                defi['lending_rates'] = {
                    'supply_apy': defi_data.get('supply_apy', 0),
                    'borrow_apy': defi_data.get('borrow_apy', 0)
                }
                
        except Exception as e:
            logger.warning(f"DeFi metrics error: {e}")
        
        return defi
    
    def _analyze_whale_activity(self, asset: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze whale trading activity"""
        whale_metrics = {
            'large_transactions': 0,
            'whale_accumulation': 0.0,
            'whale_distribution': 0.0,
            'concentration': 0.0,
            'signals': []
        }
        
        try:
            # Check recent trades for whale activity
            recent_trades = data.get('recent_trades', [])
            whale_trades = data.get('whale_trades', [])
            
            whale_metrics['large_transactions'] = len(whale_trades)
            
            # Analyze whale behavior
            if whale_trades:
                buy_volume = sum(t['amount'] for t in whale_trades if t.get('side') == 'buy')
                sell_volume = sum(t['amount'] for t in whale_trades if t.get('side') == 'sell')
                
                total_volume = buy_volume + sell_volume
                if total_volume > 0:
                    accumulation_ratio = buy_volume / total_volume
                    
                    if accumulation_ratio > 0.7:
                        whale_metrics['signals'].append(('whale_accumulation', 0.8))
                        whale_metrics['whale_accumulation'] = accumulation_ratio
                    elif accumulation_ratio < 0.3:
                        whale_metrics['signals'].append(('whale_distribution', 0.8))
                        whale_metrics['whale_distribution'] = 1 - accumulation_ratio
            
            # Address concentration (would need blockchain data)
            # For now, use a placeholder
            whale_metrics['concentration'] = 0.35  # 35% held by large addresses
            
            if whale_metrics['concentration'] > 0.5:
                whale_metrics['signals'].append(('high_concentration', 0.6))
                
        except Exception as e:
            logger.warning(f"Whale activity analysis error: {e}")
        
        return whale_metrics
    
    async def _analyze_network_health(self, asset: str) -> Dict[str, Any]:
        """Analyze blockchain network health"""
        health = {
            'hash_rate': 0.0,
            'hash_rate_change': 0.0,
            'difficulty': 0.0,
            'block_time': 0.0,
            'mempool_size': 0,
            'fee_rate': 0.0,
            'signals': []
        }
        
        try:
            # Fetch network data
            network_data = await self._fetch_network_data(asset)
            
            if network_data:
                # Hash rate (for PoW chains)
                if asset in ['btc', 'eth', 'ltc']:
                    hash_rate = network_data.get('hash_rate', 0)
                    hash_rate_change = network_data.get('hash_rate_change_30d', 0)
                    
                    health['hash_rate'] = hash_rate
                    health['hash_rate_change'] = hash_rate_change
                    
                    if hash_rate_change > 0.1:  # 10% increase
                        health['signals'].append(('increasing_security', 0.6))
                    elif hash_rate_change < -0.1:  # 10% decrease
                        health['signals'].append(('decreasing_hash_rate', 0.7))
                
                # Network congestion
                mempool_size = network_data.get('mempool_size', 0)
                avg_fee = network_data.get('avg_fee_rate', 0)
                
                health['mempool_size'] = mempool_size
                health['fee_rate'] = avg_fee
                
                # High fees indicate congestion
                if asset == 'btc' and avg_fee > 50:  # sats/vbyte
                    health['signals'].append(('network_congestion', 0.5))
                elif asset == 'eth' and avg_fee > 100:  # gwei
                    health['signals'].append(('network_congestion', 0.5))
                    
        except Exception as e:
            logger.warning(f"Network health analysis error: {e}")
        
        return health
    
    def _generate_blockchain_signals(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals from blockchain metrics"""
        all_signals = defaultdict(list)
        
        # Aggregate signals from all metric categories
        for category, data in metrics.items():
            if isinstance(data, dict) and 'signals' in data:
                for signal_name, confidence in data['signals']:
                    all_signals[signal_name].append(confidence)
        
        # Average confidence for repeated signals
        composite_signals = {}
        for signal_name, confidences in all_signals.items():
            composite_signals[signal_name] = sum(confidences) / len(confidences)
        
        return composite_signals
    
    def _calculate_confidence_and_direction(
        self, 
        signals: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> Tuple[float, Optional[str]]:
        """Calculate overall confidence and trading direction"""
        if not signals:
            return 0.0, None
        
        # Categorize signals
        bullish_signals = [
            'nvt_undervalued', 'exchange_outflow', 'decreasing_reserves',
            'whale_accumulation', 'increasing_tvl', 'network_growth',
            'mvrv_undervalued', 'low_profit_supply', 'negative_funding'
        ]
        
        bearish_signals = [
            'nvt_overvalued', 'exchange_inflow', 'increasing_reserves',
            'whale_distribution', 'decreasing_tvl', 'network_decline',
            'mvrv_overvalued', 'high_profit_supply', 'high_funding'
        ]
        
        # Calculate weighted scores
        bullish_score = sum(conf for sig, conf in signals.items() if sig in bullish_signals)
        bearish_score = sum(conf for sig, conf in signals.items() if sig in bearish_signals)
        neutral_score = sum(conf for sig, conf in signals.items() 
                          if sig not in bullish_signals and sig not in bearish_signals)
        
        total_score = bullish_score + bearish_score + neutral_score
        
        if total_score == 0:
            return 0.0, None
        
        # Determine direction
        direction = None
        if bullish_score > bearish_score * 1.3:  # 30% margin
            direction = 'long'
        elif bearish_score > bullish_score * 1.3:
            direction = 'short'
        
        # Calculate confidence
        # Higher confidence when signals agree
        signal_agreement = abs(bullish_score - bearish_score) / total_score
        base_confidence = min(total_score / 5, 1.0)  # Normalize by expected signals
        
        confidence = base_confidence * (0.7 + 0.3 * signal_agreement)
        
        return min(1.0, confidence), direction
    
    def _check_for_warnings(self, metrics: Dict[str, Any]) -> List[str]:
        """Check for any warning conditions"""
        warnings = []
        
        # High concentration risk
        whale_activity = metrics.get('whale_activity', {})
        if whale_activity.get('concentration', 0) > 0.5:
            warnings.append("High address concentration risk")
        
        # Network congestion
        network_health = metrics.get('network_health', {})
        if any('congestion' in str(s[0]) for s in network_health.get('signals', [])):
            warnings.append("Network congestion detected")
        
        # Extreme funding rates
        market_structure = metrics.get('market_structure', {})
        funding = market_structure.get('funding_rate', 0)
        if abs(funding) > 0.1:  # 10% annualized
            warnings.append(f"Extreme funding rate: {funding:.2%}")
        
        return warnings
    
    async def _fetch_onchain_data(self, asset: str) -> Optional[Dict[str, Any]]:
        """Fetch on-chain data from API"""
        # Check cache first
        cache_key = f"onchain_{asset}"
        cached = self._get_cached_data(cache_key)
        if cached:
            return cached
        
        # In production, implement actual API calls to Glassnode/CryptoQuant
        # For now, return simulated data
        simulated_data = {
            'circulating_supply': self._get_circulating_supply(asset),
            'transaction_volume_usd': np.random.uniform(1e9, 5e9),  # $1-5B daily
            'active_addresses': int(np.random.uniform(100000, 500000)),
            'address_growth_rate': np.random.uniform(-0.05, 0.1),
            'realized_cap': np.random.uniform(0.5e11, 2e11),  # $50-200B
            'supply_in_profit_pct': np.random.uniform(0.3, 0.9)
        }
        
        self._cache_data(cache_key, simulated_data)
        return simulated_data
    
    async def _fetch_exchange_flow_data(self, asset: str) -> Optional[Dict[str, Any]]:
        """Fetch exchange flow data"""
        # Check cache
        cache_key = f"exchange_flow_{asset}"
        cached = self._get_cached_data(cache_key)
        if cached:
            return cached
        
        # Simulated data
        simulated_data = {
            'inflow_volume': np.random.uniform(1000, 5000),  # BTC
            'outflow_volume': np.random.uniform(1000, 5000),
            'exchange_reserves': np.random.uniform(1e6, 2e6),
            'reserve_change_pct': np.random.uniform(-0.1, 0.1)
        }
        
        self._cache_data(cache_key, simulated_data)
        return simulated_data
    
    async def _fetch_defi_data(self, asset: str) -> Optional[Dict[str, Any]]:
        """Fetch DeFi protocol data"""
        # Check cache
        cache_key = f"defi_{asset}"
        cached = self._get_cached_data(cache_key)
        if cached:
            return cached
        
        # Simulated data
        simulated_data = {
            'tvl': np.random.uniform(1e8, 1e10),  # $100M - $10B
            'tvl_change_24h': np.random.uniform(-0.2, 0.3),
            'revenue_24h': np.random.uniform(1e5, 1e6),
            'utilization': np.random.uniform(0.4, 0.9),
            'supply_apy': np.random.uniform(0.02, 0.15),
            'borrow_apy': np.random.uniform(0.03, 0.20)
        }
        
        self._cache_data(cache_key, simulated_data)
        return simulated_data
    
    async def _fetch_network_data(self, asset: str) -> Optional[Dict[str, Any]]:
        """Fetch network health data"""
        # Check cache
        cache_key = f"network_{asset}"
        cached = self._get_cached_data(cache_key)
        if cached:
            return cached
        
        # Simulated data
        simulated_data = {
            'hash_rate': np.random.uniform(200e18, 400e18),  # EH/s for BTC
            'hash_rate_change_30d': np.random.uniform(-0.2, 0.3),
            'mempool_size': int(np.random.uniform(1000, 50000)),
            'avg_fee_rate': np.random.uniform(10, 100)  # sats/vbyte for BTC
        }
        
        self._cache_data(cache_key, simulated_data)
        return simulated_data
    
    def _get_circulating_supply(self, asset: str) -> float:
        """Get circulating supply for asset"""
        supplies = {
            'btc': 21000000,
            'eth': 120000000,
            'bnb': 150000000,
            'sol': 550000000,
            'ada': 35000000000,
            'xrp': 50000000000,
            'dot': 1300000000,
            'doge': 140000000000
        }
        return supplies.get(asset, 1000000000)
    
    def _is_defi_asset(self, asset: str) -> bool:
        """Check if asset is DeFi-related"""
        return asset.upper() in self.blockchain_config.defi_tokens
    
    def _get_cached_data(self, key: str) -> Optional[Dict[str, Any]]:
        """Get data from cache if not expired"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if (datetime.now() - timestamp).total_seconds() < self.cache_ttl:
                return data
        return None
    
    def _cache_data(self, key: str, data: Dict[str, Any]):
        """Cache data with timestamp"""
        self.cache[key] = (data, datetime.now())
    
    def is_healthy(self) -> bool:
        """Check if blockchain analyzer is healthy"""
        # Could check API connectivity here
        return True