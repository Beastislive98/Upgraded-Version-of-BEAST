"""
BEAST Trading System - Market Analyzer
Market structure, sentiment analysis, and regime detection
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone, timedelta
import logging
import re
from textblob import TextBlob
from collections import defaultdict

from config.settings import config
from utils.logger import get_logger

logger = get_logger(__name__)

class MarketAnalyzer:
    """
    Analyzes market sentiment, structure, and regime
    Combines multiple data sources for market context
    """
    
    def __init__(self, config):
        self.config = config
        self.regime_config = config.trading.regime_detection
        
        # API endpoints for sentiment data
        self.sentiment_sources = {
            'newsapi': 'https://newsapi.org/v2',
            'reddit': 'https://api.reddit.com',
            'twitter': 'https://api.twitter.com/2',  # Would need auth
            'cryptocompare': 'https://min-api.cryptocompare.com'
        }
        
        # Sentiment keywords
        self.bullish_keywords = [
            'bullish', 'moon', 'pump', 'buy', 'long', 'breakout', 'rally',
            'surge', 'soar', 'gain', 'profit', 'uptrend', 'accumulate',
            'hodl', 'diamond hands', 'to the moon', 'ath', 'all time high'
        ]
        
        self.bearish_keywords = [
            'bearish', 'dump', 'sell', 'short', 'crash', 'plunge', 'drop',
            'fall', 'decline', 'loss', 'downtrend', 'capitulation', 'fear',
            'panic', 'bubble', 'overvalued', 'correction', 'bear market'
        ]
        
        # Market regime states
        self.regime_states = {
            'bull_trend': {'min_score': 0.7, 'characteristics': ['uptrend', 'high_sentiment']},
            'bear_trend': {'min_score': 0.7, 'characteristics': ['downtrend', 'low_sentiment']},
            'ranging': {'min_score': 0.6, 'characteristics': ['sideways', 'mixed_sentiment']},
            'high_volatility': {'min_score': 0.5, 'characteristics': ['large_swings', 'uncertain']},
            'accumulation': {'min_score': 0.6, 'characteristics': ['low_volatility', 'neutral']},
            'distribution': {'min_score': 0.6, 'characteristics': ['topping', 'divergence']}
        }
        
        # Cache for API data
        self.sentiment_cache = {}
        self.cache_ttl = 600  # 10 minutes for sentiment
        
        logger.info("MarketAnalyzer initialized")
    
    async def analyze(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive market analysis
        """
        result = {
            'status': 'analyzed',
            'sentiment_score': 0.0,
            'sentiment_signals': {},
            'regime': 'unknown',
            'regime_confidence': 0.0,
            'market_structure': {},
            'correlations': {},
            'confidence': 0.0,
            'direction': None,
            'signals': {}
        }
        
        try:
            # Extract asset info
            base_asset = self._extract_base_asset(symbol)
            
            # 1. Sentiment Analysis
            sentiment_analysis = await self._analyze_sentiment(base_asset)
            result['sentiment_score'] = sentiment_analysis['score']
            result['sentiment_signals'] = sentiment_analysis['signals']
            
            # 2. Market Regime Detection
            regime_analysis = self._detect_market_regime(data, sentiment_analysis)
            result['regime'] = regime_analysis['regime']
            result['regime_confidence'] = regime_analysis['confidence']
            
            # 3. Market Structure Analysis
            structure_analysis = self._analyze_market_structure(data)
            result['market_structure'] = structure_analysis
            
            # 4. Correlation Analysis
            correlation_analysis = await self._analyze_correlations(base_asset, data)
            result['correlations'] = correlation_analysis
            
            # 5. Generate market signals
            signals = self._generate_market_signals(
                sentiment_analysis,
                regime_analysis,
                structure_analysis,
                correlation_analysis
            )
            result['signals'] = signals
            
            # 6. Calculate overall confidence and direction
            confidence, direction = self._calculate_confidence_and_direction(
                sentiment_analysis,
                regime_analysis,
                structure_analysis
            )
            result['confidence'] = confidence
            result['direction'] = direction
            
            # 7. Add market context
            result['context'] = self._get_market_context(regime_analysis, structure_analysis)
            
        except Exception as e:
            logger.error(f"Market analysis error for {symbol}: {e}")
            result['status'] = 'error'
            result['error'] = str(e)
        
        return result
    
    def _extract_base_asset(self, symbol: str) -> str:
        """Extract base asset from trading pair"""
        for separator in ['-', '/', '']:
            if separator in symbol or separator == '':
                parts = symbol.split(separator) if separator else [symbol[:3], symbol[3:]]
                if len(parts) >= 2:
                    return parts[0].upper()
        return symbol.upper()
    
    async def _analyze_sentiment(self, asset: str) -> Dict[str, Any]:
        """Analyze market sentiment from multiple sources"""
        sentiment_data = {
            'score': 0.0,
            'signals': {},
            'sources': {},
            'volume': 0,
            'trend': 'neutral'
        }
        
        try:
            # Check cache first
            cache_key = f"sentiment_{asset}"
            cached = self._get_cached_data(cache_key)
            if cached:
                return cached
            
            # Gather sentiment from multiple sources
            sentiment_tasks = {
                'news': self._fetch_news_sentiment(asset),
                'social': self._fetch_social_sentiment(asset),
                'technical': self._calculate_technical_sentiment(asset)
            }
            
            results = await asyncio.gather(*sentiment_tasks.values(), return_exceptions=True)
            
            # Process results
            valid_scores = []
            for source, result in zip(sentiment_tasks.keys(), results):
                if isinstance(result, Exception):
                    logger.warning(f"Sentiment source {source} failed: {result}")
                    continue
                
                sentiment_data['sources'][source] = result
                if 'score' in result:
                    valid_scores.append(result['score'])
            
            # Calculate composite sentiment score
            if valid_scores:
                sentiment_data['score'] = np.mean(valid_scores)
                
                # Generate sentiment signals
                if sentiment_data['score'] > 0.6:
                    sentiment_data['signals']['bullish_sentiment'] = 0.8
                    sentiment_data['trend'] = 'bullish'
                elif sentiment_data['score'] < -0.6:
                    sentiment_data['signals']['bearish_sentiment'] = 0.8
                    sentiment_data['trend'] = 'bearish'
                else:
                    sentiment_data['signals']['neutral_sentiment'] = 0.6
                    sentiment_data['trend'] = 'neutral'
            
            # Cache results
            self._cache_data(cache_key, sentiment_data)
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
        
        return sentiment_data
    
    async def _fetch_news_sentiment(self, asset: str) -> Dict[str, Any]:
        """Fetch and analyze news sentiment"""
        try:
            # In production, implement actual news API calls
            # For now, simulate with realistic data
            
            # Simulate fetching news articles
            articles = [
                f"{asset} shows strong momentum amid institutional interest",
                f"Analysts predict continued growth for {asset}",
                f"Market concerns over {asset} regulatory challenges",
                f"{asset} breaks key resistance level"
            ]
            
            # Analyze sentiment of articles
            sentiments = []
            for article in articles:
                blob = TextBlob(article)
                sentiments.append(blob.sentiment.polarity)
            
            # Count keyword occurrences
            all_text = ' '.join(articles).lower()
            bullish_count = sum(1 for keyword in self.bullish_keywords if keyword in all_text)
            bearish_count = sum(1 for keyword in self.bearish_keywords if keyword in all_text)
            
            # Calculate news sentiment score
            text_sentiment = np.mean(sentiments) if sentiments else 0
            keyword_sentiment = (bullish_count - bearish_count) / max(bullish_count + bearish_count, 1)
            
            news_score = 0.7 * text_sentiment + 0.3 * keyword_sentiment
            
            return {
                'score': np.clip(news_score, -1, 1),
                'article_count': len(articles),
                'bullish_keywords': bullish_count,
                'bearish_keywords': bearish_count
            }
            
        except Exception as e:
            logger.warning(f"News sentiment error: {e}")
            return {'score': 0.0, 'error': str(e)}
    
    async def _fetch_social_sentiment(self, asset: str) -> Dict[str, Any]:
        """Fetch and analyze social media sentiment"""
        try:
            # In production, implement actual social media API calls
            # For now, simulate with realistic patterns
            
            # Simulate social media metrics
            mentions = np.random.randint(1000, 10000)
            positive_ratio = np.random.beta(6, 4)  # Slightly positive bias
            
            # Simulate trending status
            is_trending = np.random.random() > 0.7
            
            # Calculate social sentiment score
            base_score = (positive_ratio - 0.5) * 2  # Convert to -1 to 1 range
            
            # Boost score if trending
            if is_trending:
                base_score *= 1.2
            
            # Volume factor (more mentions = stronger signal)
            volume_factor = min(mentions / 5000, 1.0)
            final_score = base_score * (0.5 + 0.5 * volume_factor)
            
            return {
                'score': np.clip(final_score, -1, 1),
                'mentions': mentions,
                'positive_ratio': positive_ratio,
                'is_trending': is_trending
            }
            
        except Exception as e:
            logger.warning(f"Social sentiment error: {e}")
            return {'score': 0.0, 'error': str(e)}
    
    async def _calculate_technical_sentiment(self, asset: str) -> Dict[str, Any]:
        """Calculate sentiment from technical indicators"""
        try:
            # This would use technical indicators to gauge market sentiment
            # For now, return a neutral score
            return {
                'score': 0.0,
                'rsi_sentiment': 'neutral',
                'momentum_sentiment': 'neutral'
            }
            
        except Exception as e:
            logger.warning(f"Technical sentiment error: {e}")
            return {'score': 0.0, 'error': str(e)}
    
    def _detect_market_regime(
        self, 
        data: Dict[str, Any],
        sentiment_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect current market regime"""
        regime_result = {
            'regime': 'unknown',
            'confidence': 0.0,
            'characteristics': [],
            'scores': {}
        }
        
        try:
            price_data = data.get('price_data')
            if price_data is None or len(price_data) < self.regime_config['lookback_periods']:
                return regime_result
            
            # Calculate regime indicators
            indicators = self._calculate_regime_indicators(price_data)
            
            # Score each regime
            regime_scores = {}
            
            # Bull trend
            bull_score = 0.0
            if indicators['trend_strength'] > 0 and indicators['trend_direction'] == 'up':
                bull_score += 0.4
            if sentiment_analysis['score'] > 0.3:
                bull_score += 0.3
            if indicators['higher_highs'] and indicators['higher_lows']:
                bull_score += 0.3
            regime_scores['bull_trend'] = bull_score
            
            # Bear trend
            bear_score = 0.0
            if indicators['trend_strength'] < 0 and indicators['trend_direction'] == 'down':
                bear_score += 0.4
            if sentiment_analysis['score'] < -0.3:
                bear_score += 0.3
            if indicators['lower_highs'] and indicators['lower_lows']:
                bear_score += 0.3
            regime_scores['bear_trend'] = bear_score
            
            # Ranging market
            range_score = 0.0
            if abs(indicators['trend_strength']) < 0.1:
                range_score += 0.4
            if indicators['volatility'] < indicators['avg_volatility']:
                range_score += 0.3
            if not (indicators['higher_highs'] or indicators['lower_lows']):
                range_score += 0.3
            regime_scores['ranging'] = range_score
            
            # High volatility
            vol_score = 0.0
            if indicators['volatility'] > indicators['avg_volatility'] * 1.5:
                vol_score += 0.5
            if indicators['volatility_trend'] == 'increasing':
                vol_score += 0.3
            if abs(sentiment_analysis['score']) < 0.2:  # Uncertainty
                vol_score += 0.2
            regime_scores['high_volatility'] = vol_score
            
            # Accumulation
            accum_score = 0.0
            if indicators['volume_trend'] == 'increasing' and abs(indicators['trend_strength']) < 0.1:
                accum_score += 0.4
            if indicators['volatility'] < indicators['avg_volatility'] * 0.7:
                accum_score += 0.3
            if indicators['support_tests'] > 2:
                accum_score += 0.3
            regime_scores['accumulation'] = accum_score
            
            # Distribution
            dist_score = 0.0
            if indicators['volume_trend'] == 'decreasing' and indicators['trend_direction'] == 'up':
                dist_score += 0.4
            if indicators['resistance_tests'] > 2:
                dist_score += 0.3
            if sentiment_analysis['score'] > 0.7 and indicators['momentum'] < 0:
                dist_score += 0.3
            regime_scores['distribution'] = dist_score
            
            # Select regime with highest score
            best_regime = max(regime_scores.items(), key=lambda x: x[1])
            regime_name, regime_score = best_regime
            
            # Check if score meets minimum threshold
            min_score = self.regime_states[regime_name]['min_score']
            if regime_score >= min_score:
                regime_result['regime'] = regime_name
                regime_result['confidence'] = regime_score
                regime_result['characteristics'] = self.regime_states[regime_name]['characteristics']
            else:
                regime_result['regime'] = 'uncertain'
                regime_result['confidence'] = 0.5
            
            regime_result['scores'] = regime_scores
            
        except Exception as e:
            logger.error(f"Regime detection error: {e}")
        
        return regime_result
    
    def _calculate_regime_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate indicators for regime detection"""
        indicators = {
            'trend_strength': 0.0,
            'trend_direction': 'neutral',
            'volatility': 0.0,
            'avg_volatility': 0.0,
            'volatility_trend': 'stable',
            'volume_trend': 'stable',
            'momentum': 0.0,
            'higher_highs': False,
            'higher_lows': False,
            'lower_highs': False,
            'lower_lows': False,
            'support_tests': 0,
            'resistance_tests': 0
        }
        
        try:
            lookback = self.regime_config['lookback_periods']
            
            # Trend analysis
            prices = df['close'].values[-lookback:]
            x = np.arange(len(prices))
            slope, _ = np.polyfit(x, prices, 1)
            
            # Normalize slope
            price_range = prices.max() - prices.min()
            if price_range > 0:
                indicators['trend_strength'] = slope / price_range
            
            # Trend direction
            if indicators['trend_strength'] > self.regime_config['trend_threshold']:
                indicators['trend_direction'] = 'up'
            elif indicators['trend_strength'] < -self.regime_config['trend_threshold']:
                indicators['trend_direction'] = 'down'
            
            # Volatility
            returns = df['close'].pct_change().dropna()
            indicators['volatility'] = returns.iloc[-lookback:].std()
            indicators['avg_volatility'] = returns.rolling(window=lookback*2).std().mean()
            
            # Volatility trend
            recent_vol = returns.iloc[-lookback//2:].std()
            older_vol = returns.iloc[-lookback:-lookback//2].std()
            if recent_vol > older_vol * 1.2:
                indicators['volatility_trend'] = 'increasing'
            elif recent_vol < older_vol * 0.8:
                indicators['volatility_trend'] = 'decreasing'
            
            # Volume trend
            if 'volume' in df.columns:
                recent_volume = df['volume'].iloc[-lookback//2:].mean()
                older_volume = df['volume'].iloc[-lookback:-lookback//2].mean()
                if recent_volume > older_volume * 1.2:
                    indicators['volume_trend'] = 'increasing'
                elif recent_volume < older_volume * 0.8:
                    indicators['volume_trend'] = 'decreasing'
            
            # Momentum
            indicators['momentum'] = (prices[-1] - prices[-lookback//2]) / prices[-lookback//2]
            
            # Price structure
            highs = df['high'].values[-lookback:]
            lows = df['low'].values[-lookback:]
            
            # Check for higher highs/lows or lower highs/lows
            mid_point = lookback // 2
            indicators['higher_highs'] = highs[-1] > highs[:mid_point].max()
            indicators['higher_lows'] = lows[-1] > lows[:mid_point].min()
            indicators['lower_highs'] = highs[-1] < highs[:mid_point].max()
            indicators['lower_lows'] = lows[-1] < lows[:mid_point].min()
            
            # Support/Resistance tests
            current_price = prices[-1]
            price_levels = np.percentile(prices, [25, 50, 75])
            
            # Count touches near support/resistance
            for price in prices[-10:]:
                if abs(price - price_levels[0]) / price_levels[0] < 0.02:  # 2% tolerance
                    indicators['support_tests'] += 1
                if abs(price - price_levels[2]) / price_levels[2] < 0.02:
                    indicators['resistance_tests'] += 1
                    
        except Exception as e:
            logger.warning(f"Regime indicator calculation error: {e}")
        
        return indicators
    
    def _analyze_market_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market microstructure"""
        structure = {
            'spread': 0.0,
            'spread_volatility': 0.0,
            'order_book_imbalance': 0.0,
            'trade_size_distribution': 'normal',
            'price_impact': 0.0,
            'liquidity_score': 0.5
        }
        
        try:
            # Analyze order book if available
            orderbook = data.get('orderbook', {})
            if orderbook:
                # Spread analysis
                if 'spread_pct' in orderbook:
                    structure['spread'] = orderbook['spread_pct']
                
                # Order book imbalance
                if 'order_flow_imbalance' in orderbook:
                    structure['order_book_imbalance'] = orderbook['order_flow_imbalance']
                
                # Liquidity score based on depth
                bid_volume = orderbook.get('bid_volume', 0)
                ask_volume = orderbook.get('ask_volume', 0)
                total_volume = bid_volume + ask_volume
                
                if total_volume > 0:
                    # Higher volume = better liquidity
                    structure['liquidity_score'] = min(1.0, total_volume / 1000)  # Normalize
            
            # Analyze recent trades if available
            recent_trades = data.get('recent_trades', [])
            if recent_trades:
                trade_sizes = [t['amount'] for t in recent_trades]
                if trade_sizes:
                    # Check for unusual trade sizes
                    avg_size = np.mean(trade_sizes)
                    std_size = np.std(trade_sizes)
                    large_trades = sum(1 for size in trade_sizes if size > avg_size + 2*std_size)
                    
                    if large_trades > len(trade_sizes) * 0.1:  # More than 10% large trades
                        structure['trade_size_distribution'] = 'heavy_tail'
                    else:
                        structure['trade_size_distribution'] = 'normal'
                        
        except Exception as e:
            logger.warning(f"Market structure analysis error: {e}")
        
        return structure
    
    async def _analyze_correlations(self, asset: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze correlations with other assets"""
        correlations = {
            'btc_correlation': 0.0,
            'market_beta': 1.0,
            'sector_correlation': 0.0,
            'correlation_trend': 'stable'
        }
        
        try:
            # In production, fetch correlation data
            # For now, simulate realistic correlations
            
            if asset != 'BTC':
                # Most cryptos have positive correlation with BTC
                correlations['btc_correlation'] = np.random.uniform(0.5, 0.9)
            else:
                correlations['btc_correlation'] = 1.0
            
            # Market beta (sensitivity to overall market moves)
            if asset in ['ETH', 'BNB']:
                correlations['market_beta'] = np.random.uniform(1.1, 1.3)
            elif asset in ['BTC']:
                correlations['market_beta'] = 1.0
            else:
                correlations['market_beta'] = np.random.uniform(1.2, 1.8)
            
            # Sector correlation (e.g., DeFi tokens correlate with each other)
            correlations['sector_correlation'] = np.random.uniform(0.6, 0.8)
            
            # Correlation trend
            trend_random = np.random.random()
            if trend_random > 0.7:
                correlations['correlation_trend'] = 'increasing'
            elif trend_random < 0.3:
                correlations['correlation_trend'] = 'decreasing'
                
        except Exception as e:
            logger.warning(f"Correlation analysis error: {e}")
        
        return correlations
    
    def _generate_market_signals(
        self,
        sentiment: Dict[str, Any],
        regime: Dict[str, Any],
        structure: Dict[str, Any],
        correlations: Dict[str, Any]
    ) -> Dict[str, float]:
        """Generate trading signals from market analysis"""
        signals = {}
        
        # Sentiment signals
        if sentiment['score'] > 0.6:
            signals['positive_sentiment'] = min(1.0, sentiment['score'])
        elif sentiment['score'] < -0.6:
            signals['negative_sentiment'] = min(1.0, abs(sentiment['score']))
        
        # Regime signals
        regime_name = regime['regime']
        regime_confidence = regime['confidence']
        
        if regime_name == 'bull_trend' and regime_confidence > 0.7:
            signals['bull_regime'] = regime_confidence
        elif regime_name == 'bear_trend' and regime_confidence > 0.7:
            signals['bear_regime'] = regime_confidence
        elif regime_name == 'high_volatility':
            signals['high_volatility_regime'] = regime_confidence
        elif regime_name == 'accumulation':
            signals['accumulation_phase'] = regime_confidence
        elif regime_name == 'distribution':
            signals['distribution_phase'] = regime_confidence
        
        # Structure signals
        if structure['order_book_imbalance'] > 0.3:
            signals['bullish_orderflow'] = structure['order_book_imbalance']
        elif structure['order_book_imbalance'] < -0.3:
            signals['bearish_orderflow'] = abs(structure['order_book_imbalance'])
        
        if structure['liquidity_score'] < 0.3:
            signals['low_liquidity'] = 1.0 - structure['liquidity_score']
        
        # Correlation signals
        if correlations['btc_correlation'] > 0.8 and correlations['correlation_trend'] == 'increasing':
            signals['high_correlation'] = correlations['btc_correlation']
        
        if correlations['market_beta'] > 1.5:
            signals['high_beta'] = min(1.0, correlations['market_beta'] / 2)
        
        return signals
    
    def _calculate_confidence_and_direction(
        self,
        sentiment: Dict[str, Any],
        regime: Dict[str, Any],
        structure: Dict[str, Any]
    ) -> Tuple[float, Optional[str]]:
        """Calculate overall market confidence and direction"""
        # Base confidence from regime confidence
        confidence = regime['confidence'] * 0.5
        
        # Add sentiment contribution
        sentiment_strength = abs(sentiment['score'])
        confidence += sentiment_strength * 0.3
        
        # Add structure contribution
        liquidity = structure['liquidity_score']
        confidence += liquidity * 0.2
        
        # Determine direction
        direction = None
        
        if regime['regime'] in ['bull_trend', 'accumulation']:
            if sentiment['score'] > 0:
                direction = 'long'
        elif regime['regime'] in ['bear_trend', 'distribution']:
            if sentiment['score'] < 0:
                direction = 'short'
        elif regime['regime'] == 'ranging':
            # In ranging market, follow sentiment
            if sentiment['score'] > 0.4:
                direction = 'long'
            elif sentiment['score'] < -0.4:
                direction = 'short'
        
        # Cap confidence at 1.0
        confidence = min(1.0, confidence)
        
        # Reduce confidence if direction conflicts with regime
        if direction == 'long' and regime['regime'] == 'bear_trend':
            confidence *= 0.7
        elif direction == 'short' and regime['regime'] == 'bull_trend':
            confidence *= 0.7
        
        return confidence, direction
    
    def _get_market_context(
        self,
        regime: Dict[str, Any],
        structure: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get additional market context"""
        context = {
            'trading_environment': 'normal',
            'recommended_strategy_type': 'trend_following',
            'risk_adjustment': 1.0,
            'notes': []
        }
        
        # Determine trading environment
        if regime['regime'] == 'high_volatility':
            context['trading_environment'] = 'volatile'
            context['risk_adjustment'] = 0.7
            context['notes'].append('High volatility - reduce position sizes')
        elif structure['liquidity_score'] < 0.3:
            context['trading_environment'] = 'illiquid'
            context['risk_adjustment'] = 0.5
            context['notes'].append('Low liquidity - use limit orders')
        
        # Recommend strategy type based on regime
        if regime['regime'] in ['bull_trend', 'bear_trend']:
            context['recommended_strategy_type'] = 'trend_following'
        elif regime['regime'] == 'ranging':
            context['recommended_strategy_type'] = 'mean_reversion'
        elif regime['regime'] == 'high_volatility':
            context['recommended_strategy_type'] = 'volatility'
        elif regime['regime'] == 'accumulation':
            context['recommended_strategy_type'] = 'accumulation'
        
        return context
    
    def _get_cached_data(self, key: str) -> Optional[Dict[str, Any]]:
        """Get data from cache if not expired"""
        if key in self.sentiment_cache:
            data, timestamp = self.sentiment_cache[key]
            if (datetime.now() - timestamp).total_seconds() < self.cache_ttl:
                return data
        return None
    
    def _cache_data(self, key: str, data: Dict[str, Any]):
        """Cache data with timestamp"""
        self.sentiment_cache[key] = (data, datetime.now())
    
    def is_healthy(self) -> bool:
        """Check if market analyzer is healthy"""
        # Could check API connectivity here
        return True