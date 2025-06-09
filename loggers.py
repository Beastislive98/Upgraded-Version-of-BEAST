"""
BEAST Trading System - Logger Utility
Centralized logging with structured output and performance tracking
"""

import logging
import sys
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import traceback
from functools import wraps
import time
from collections import deque
import threading
from enum import Enum

# Ensure logs directory exists
Path("logs").mkdir(exist_ok=True)
Path("logs/trades").mkdir(exist_ok=True)
Path("logs/decisions").mkdir(exist_ok=True)
Path("logs/performance").mkdir(exist_ok=True)
Path("logs/errors").mkdir(exist_ok=True)

class LogLevel(Enum):
    """Log levels with colors for console output"""
    DEBUG = ("DEBUG", "\033[36m")      # Cyan
    INFO = ("INFO", "\033[32m")        # Green
    WARNING = ("WARNING", "\033[33m")  # Yellow
    ERROR = ("ERROR", "\033[31m")      # Red
    CRITICAL = ("CRITICAL", "\033[35m") # Magenta

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging"""
    
    def __init__(self, use_color: bool = True):
        super().__init__()
        self.use_color = use_color and sys.stderr.isatty()
        
    def format(self, record: logging.LogRecord) -> str:
        # Base log structure
        log_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': record.levelname,
            'module': record.name,
            'message': record.getMessage(),
            'thread': threading.current_thread().name
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Format based on output type
        if self.use_color:
            # Console output with color
            level_color = self._get_level_color(record.levelname)
            reset_color = "\033[0m"
            
            return (f"{level_color}[{log_data['timestamp']}] "
                   f"{log_data['level']:<8}{reset_color} "
                   f"{log_data['module']:<20} | "
                   f"{log_data['message']}")
        else:
            # File output as JSON
            return json.dumps(log_data, default=str)
    
    def _get_level_color(self, level: str) -> str:
        """Get color code for log level"""
        for log_level in LogLevel:
            if log_level.value[0] == level:
                return log_level.value[1]
        return ""

class PerformanceLogger:
    """Logger for performance metrics"""
    
    def __init__(self, window_size: int = 1000):
        self.metrics = deque(maxlen=window_size)
        self.logger = logging.getLogger('performance')
        
    def log_execution_time(self, operation: str, duration: float, metadata: Dict[str, Any] = None):
        """Log execution time for an operation"""
        metric = {
            'timestamp': datetime.now(timezone.utc),
            'operation': operation,
            'duration_ms': duration * 1000,
            'metadata': metadata or {}
        }
        
        self.metrics.append(metric)
        
        # Log to file
        self.logger.info(
            f"Performance: {operation} took {duration*1000:.2f}ms",
            extra={'extra_data': metric}
        )
    
    def get_stats(self, operation: Optional[str] = None) -> Dict[str, float]:
        """Get performance statistics"""
        if operation:
            relevant_metrics = [m for m in self.metrics if m['operation'] == operation]
        else:
            relevant_metrics = list(self.metrics)
        
        if not relevant_metrics:
            return {}
        
        durations = [m['duration_ms'] for m in relevant_metrics]
        
        return {
            'count': len(durations),
            'avg_ms': sum(durations) / len(durations),
            'min_ms': min(durations),
            'max_ms': max(durations),
            'p50_ms': sorted(durations)[len(durations)//2],
            'p95_ms': sorted(durations)[int(len(durations)*0.95)] if len(durations) > 20 else max(durations)
        }

class BeastLogger:
    """Main logger class for BEAST system"""
    
    _instances = {}
    _lock = threading.Lock()
    
    def __new__(cls, name: str):
        # Singleton pattern per logger name
        with cls._lock:
            if name not in cls._instances:
                cls._instances[name] = super().__new__(cls)
            return cls._instances[name]
    
    def __init__(self, name: str):
        if hasattr(self, '_initialized'):
            return
            
        self.name = name
        self.logger = logging.getLogger(name)
        self.performance = PerformanceLogger()
        
        # Setup handlers if not already configured
        if not self.logger.handlers:
            self._setup_handlers()
        
        self._initialized = True
    
    def _setup_handlers(self):
        """Setup logging handlers"""
        # Console handler with color
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(StructuredFormatter(use_color=True))
        console_handler.setLevel(logging.INFO)
        
        # Main file handler (JSON format)
        file_handler = logging.FileHandler(f'logs/beast_{datetime.now().strftime("%Y%m%d")}.log')
        file_handler.setFormatter(StructuredFormatter(use_color=False))
        file_handler.setLevel(logging.DEBUG)
        
        # Error file handler
        error_handler = logging.FileHandler('logs/errors/errors.log')
        error_handler.setFormatter(StructuredFormatter(use_color=False))
        error_handler.setLevel(logging.ERROR)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
        
        # Set level
        self.logger.setLevel(logging.DEBUG)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, extra={'extra_data': kwargs})
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, extra={'extra_data': kwargs})
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, extra={'extra_data': kwargs})
    
    def error(self, message: str, exc_info: bool = True, **kwargs):
        """Log error message"""
        self.logger.error(message, exc_info=exc_info, extra={'extra_data': kwargs})
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(message, extra={'extra_data': kwargs})
    
    def trade(self, trade_data: Dict[str, Any]):
        """Log trade execution"""
        trade_logger = logging.getLogger('trades')
        if not trade_logger.handlers:
            handler = logging.FileHandler(f'logs/trades/trades_{datetime.now().strftime("%Y%m%d")}.jsonl')
            handler.setFormatter(StructuredFormatter(use_color=False))
            trade_logger.addHandler(handler)
            trade_logger.setLevel(logging.INFO)
        
        trade_logger.info("Trade executed", extra={'extra_data': trade_data})
    
    def decision(self, decision_data: Dict[str, Any]):
        """Log trading decision"""
        decision_logger = logging.getLogger('decisions')
        if not decision_logger.handlers:
            handler = logging.FileHandler('logs/decisions/trade_decisions.jsonl')
            handler.setFormatter(StructuredFormatter(use_color=False))
            decision_logger.addHandler(handler)
            decision_logger.setLevel(logging.INFO)
        
        decision_logger.info("Decision made", extra={'extra_data': decision_data})

# Performance tracking decorator
def track_performance(operation: str = None):
    """Decorator to track function performance"""
    def decorator(func):
        op_name = operation or f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger = get_logger(func.__module__)
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Log performance
                logger.performance.log_execution_time(
                    op_name,
                    duration,
                    {'args_count': len(args), 'kwargs_count': len(kwargs)}
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"Error in {op_name} after {duration:.2f}s: {str(e)}",
                    operation=op_name,
                    duration=duration
                )
                raise
        
        return wrapper
    return decorator

# Global logger instance cache
_logger_cache = {}

def get_logger(name: str) -> BeastLogger:
    """Get or create a logger instance"""
    if name not in _logger_cache:
        _logger_cache[name] = BeastLogger(name)
    return _logger_cache[name]

# Convenience functions for specific logging needs
def log_trade_decision(symbol: str, decision: str, confidence: float, reasoning: Dict[str, Any]):
    """Log a trade decision"""
    logger = get_logger('decisions')
    logger.decision({
        'symbol': symbol,
        'decision': decision,
        'confidence': confidence,
        'reasoning': reasoning,
        'timestamp': datetime.now(timezone.utc)
    })

def log_trade_execution(
    symbol: str,
    side: str,
    size: float,
    price: float,
    order_id: str,
    strategy: str,
    metadata: Dict[str, Any] = None
):
    """Log a trade execution"""
    logger = get_logger('trades')
    logger.trade({
        'symbol': symbol,
        'side': side,
        'size': size,
        'price': price,
        'order_id': order_id,
        'strategy': strategy,
        'metadata': metadata or {},
        'timestamp': datetime.now(timezone.utc)
    })

def log_performance(metrics: Dict[str, Any]):
    """Log performance metrics"""
    perf_logger = logging.getLogger('performance')
    if not perf_logger.handlers:
        handler = logging.FileHandler(f'logs/performance/performance_{datetime.now().strftime("%Y%m%d")}.jsonl')
        handler.setFormatter(StructuredFormatter(use_color=False))
        perf_logger.addHandler(handler)
        perf_logger.setLevel(logging.INFO)
    
    perf_logger.info("Performance metrics", extra={'extra_data': metrics})

def log_error(module: str, error: Exception, context: Dict[str, Any] = None):
    """Log an error with context"""
    logger = get_logger(module)
    logger.error(
        f"Error in {module}: {str(error)}",
        exc_info=True,
        error_type=type(error).__name__,
        context=context or {}
    )

# Initialize root logger
root_logger = get_logger('beast')
root_logger.info("BEAST Trading System logger initialized")