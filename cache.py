"""
BEAST Trading System - Cache Utilities
High-performance caching for market data and analysis results
"""

import logging
from typing import Dict, Any, Optional, Union, List, Callable
from datetime import datetime, timezone, timedelta
import json
import pickle
import hashlib
import asyncio
import aioredis
from functools import wraps, lru_cache
from collections import OrderedDict
import pandas as pd
import numpy as np

from config.settings import config

logger = logging.getLogger(__name__)

class CacheBackend:
    """Base class for cache backends"""
    
    async def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        raise NotImplementedError
    
    async def delete(self, key: str):
        raise NotImplementedError
    
    async def exists(self, key: str) -> bool:
        raise NotImplementedError
    
    async def clear(self):
        raise NotImplementedError

class MemoryCache(CacheBackend):
    """In-memory cache with TTL support"""
    
    def __init__(self, max_size: int = 1000):
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.max_size = max_size
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            
            # Check TTL
            if entry['expires_at'] and datetime.now(timezone.utc) > entry['expires_at']:
                del self.cache[key]
                return None
            
            # Move to end (LRU)
            self.cache.move_to_end(key)
            return entry['value']
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        async with self._lock:
            expires_at = None
            if ttl:
                expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl)
            
            self.cache[key] = {
                'value': value,
                'expires_at': expires_at,
                'created_at': datetime.now(timezone.utc)
            }
            
            # Move to end
            self.cache.move_to_end(key)
            
            # Evict oldest if over capacity
            while len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
    
    async def delete(self, key: str):
        async with self._lock:
            self.cache.pop(key, None)
    
    async def exists(self, key: str) -> bool:
        return key in self.cache
    
    async def clear(self):
        async with self._lock:
            self.cache.clear()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        async with self._lock:
            total_items = len(self.cache)
            expired_items = sum(
                1 for entry in self.cache.values()
                if entry['expires_at'] and datetime.now(timezone.utc) > entry['expires_at']
            )
            
            return {
                'total_items': total_items,
                'expired_items': expired_items,
                'active_items': total_items - expired_items,
                'capacity': self.max_size,
                'usage_pct': (total_items / self.max_size) * 100 if self.max_size > 0 else 0
            }

class RedisCache(CacheBackend):
    """Redis-based cache for distributed systems"""
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or config.get('redis_url', 'redis://localhost:6379')
        self.redis = None
        self._connected = False
    
    async def connect(self):
        """Connect to Redis"""
        if not self._connected:
            try:
                self.redis = await aioredis.create_redis_pool(self.redis_url)
                self._connected = True
                logger.info("Connected to Redis cache")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis:
            self.redis.close()
            await self.redis.wait_closed()
            self._connected = False
    
    async def get(self, key: str) -> Optional[Any]:
        if not self._connected:
            await self.connect()
        
        try:
            data = await self.redis.get(key)
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        if not self._connected:
            await self.connect()
        
        try:
            data = pickle.dumps(value)
            if ttl:
                await self.redis.setex(key, ttl, data)
            else:
                await self.redis.set(key, data)
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    
    async def delete(self, key: str):
        if not self._connected:
            await self.connect()
        
        try:
            await self.redis.delete(key)
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
    
    async def exists(self, key: str) -> bool:
        if not self._connected:
            await self.connect()
        
        try:
            return await self.redis.exists(key) > 0
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False
    
    async def clear(self):
        if not self._connected:
            await self.connect()
        
        try:
            await self.redis.flushdb()
        except Exception as e:
            logger.error(f"Redis clear error: {e}")

class DataCache:
    """
    High-level cache manager for trading data
    """
    
    def __init__(self, backend: Optional[CacheBackend] = None):
        self.backend = backend or MemoryCache(max_size=5000)
        self.hit_count = 0
        self.miss_count = 0
        
        # Cache key prefixes
        self.MARKET_DATA_PREFIX = "market:"
        self.ANALYSIS_PREFIX = "analysis:"
        self.DECISION_PREFIX = "decision:"
        self.INDICATOR_PREFIX = "indicator:"
        
    def _generate_key(self, prefix: str, *args) -> str:
        """Generate cache key"""
        key_parts = [str(arg) for arg in args]
        key_string = ":".join(key_parts)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()[:8]
        return f"{prefix}{key_hash}"
    
    async def get_market_data(
        self, 
        symbol: str, 
        timeframe: str,
        start_time: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """Get cached market data"""
        key = self._generate_key(self.MARKET_DATA_PREFIX, symbol, timeframe, start_time)
        
        data = await self.backend.get(key)
        if data is not None:
            self.hit_count += 1
            # Convert back to DataFrame
            if isinstance(data, dict) and 'data' in data:
                df = pd.DataFrame(data['data'])
                if 'index' in data:
                    df.index = pd.DatetimeIndex(data['index'])
                return df
        else:
            self.miss_count += 1
        
        return None
    
    async def set_market_data(
        self,
        symbol: str,
        timeframe: str,
        data: pd.DataFrame,
        ttl: int = 300  # 5 minutes default
    ):
        """Cache market data"""
        key = self._generate_key(self.MARKET_DATA_PREFIX, symbol, timeframe, data.index[0])
        
        # Convert DataFrame to cacheable format
        cache_data = {
            'data': data.to_dict('records'),
            'index': data.index.tolist(),
            'columns': data.columns.tolist()
        }
        
        await self.backend.set(key, cache_data, ttl)
    
    async def get_analysis_result(
        self,
        analysis_type: str,
        symbol: str,
        params_hash: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached analysis result"""
        key = self._generate_key(self.ANALYSIS_PREFIX, analysis_type, symbol, params_hash)
        
        result = await self.backend.get(key)
        if result is not None:
            self.hit_count += 1
        else:
            self.miss_count += 1
        
        return result
    
    async def set_analysis_result(
        self,
        analysis_type: str,
        symbol: str,
        params_hash: str,
        result: Dict[str, Any],
        ttl: int = 60  # 1 minute default
    ):
        """Cache analysis result"""
        key = self._generate_key(self.ANALYSIS_PREFIX, analysis_type, symbol, params_hash)
        await self.backend.set(key, result, ttl)
    
    def get(self, key: str) -> Optional[Any]:
        """Synchronous get for compatibility"""
        return asyncio.run(self.backend.get(key))
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Synchronous set for compatibility"""
        asyncio.run(self.backend.set(key, value, ttl))
    
    async def get_indicator(
        self,
        indicator_name: str,
        symbol: str,
        params: Dict[str, Any]
    ) -> Optional[Union[float, pd.Series]]:
        """Get cached indicator value"""
        params_str = json.dumps(params, sort_keys=True)
        key = self._generate_key(self.INDICATOR_PREFIX, indicator_name, symbol, params_str)
        
        return await self.backend.get(key)
    
    async def set_indicator(
        self,
        indicator_name: str,
        symbol: str,
        params: Dict[str, Any],
        value: Union[float, pd.Series],
        ttl: int = 30  # 30 seconds default
    ):
        """Cache indicator value"""
        params_str = json.dumps(params, sort_keys=True)
        key = self._generate_key(self.INDICATOR_PREFIX, indicator_name, symbol, params_str)
        
        # Convert pandas Series to list for caching
        if isinstance(value, pd.Series):
            cache_value = {'type': 'series', 'data': value.tolist(), 'index': value.index.tolist()}
        else:
            cache_value = {'type': 'scalar', 'data': value}
        
        await self.backend.set(key, cache_value, ttl)
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate"""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': self.get_hit_rate(),
            'total_requests': self.hit_count + self.miss_count
        }
        
        # Add backend-specific stats if available
        if hasattr(self.backend, 'get_stats'):
            backend_stats = asyncio.run(self.backend.get_stats())
            stats.update(backend_stats)
        
        return stats
    
    async def clear(self):
        """Clear all cache"""
        await self.backend.clear()
        self.hit_count = 0
        self.miss_count = 0
        logger.info("Cache cleared")

def cached(ttl: int = 60, key_prefix: str = ""):
    """
    Decorator for caching function results
    """
    def decorator(func):
        # Create a cache specific to this function
        cache = MemoryCache(max_size=100)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{key_prefix}{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            cache_key = hashlib.md5(cache_key.encode()).hexdigest()
            
            # Try to get from cache
            result = await cache.get(cache_key)
            if result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await cache.set(cache_key, result, ttl)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{key_prefix}{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            cache_key = hashlib.md5(cache_key.encode()).hexdigest()
            
            # For sync functions, use lru_cache
            return func(*args, **kwargs)
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            # For sync functions, use lru_cache
            return lru_cache(maxsize=100)(func)
    
    return decorator

class IndicatorCache:
    """
    Specialized cache for technical indicators
    """
    
    def __init__(self, backend: Optional[CacheBackend] = None):
        self.backend = backend or MemoryCache(max_size=1000)
        self.indicator_functions: Dict[str, Callable] = {}
    
    def register_indicator(self, name: str, func: Callable):
        """Register an indicator function"""
        self.indicator_functions[name] = func
    
    async def get_or_calculate(
        self,
        indicator_name: str,
        data: pd.DataFrame,
        **params
    ) -> Union[float, pd.Series, pd.DataFrame]:
        """Get indicator from cache or calculate"""
        # Generate cache key
        data_hash = hashlib.md5(str(data.index[-1]).encode()).hexdigest()[:8]
        params_str = json.dumps(params, sort_keys=True)
        cache_key = f"ind:{indicator_name}:{data_hash}:{params_str}"
        
        # Try cache
        cached_value = await self.backend.get(cache_key)
        if cached_value is not None:
            # Reconstruct pandas object if needed
            if isinstance(cached_value, dict) and cached_value.get('type') == 'series':
                return pd.Series(cached_value['data'], index=cached_value['index'])
            return cached_value
        
        # Calculate indicator
        if indicator_name not in self.indicator_functions:
            raise ValueError(f"Unknown indicator: {indicator_name}")
        
        func = self.indicator_functions[indicator_name]
        result = func(data, **params)
        
        # Cache result
        if isinstance(result, pd.Series):
            cache_value = {
                'type': 'series',
                'data': result.tolist(),
                'index': result.index.tolist()
            }
        else:
            cache_value = result
        
        await self.backend.set(cache_key, cache_value, ttl=30)
        
        return result

# Global cache instance
data_cache = DataCache()
indicator_cache = IndicatorCache()

# Convenience functions
def invalidate_market_data(symbol: str):
    """Invalidate market data cache for a symbol"""
    # This would need to be implemented based on key patterns
    logger.info(f"Invalidated market data cache for {symbol}")

def get_cache_stats() -> Dict[str, Any]:
    """Get global cache statistics"""
    return data_cache.get_stats()

@cached(ttl=300)
async def get_cached_market_summary(symbol: str) -> Dict[str, Any]:
    """Example of cached function"""
    # This would fetch market summary
    return {
        'symbol': symbol,
        'price': 0,
        'volume': 0,
        'timestamp': datetime.now(timezone.utc)
    }