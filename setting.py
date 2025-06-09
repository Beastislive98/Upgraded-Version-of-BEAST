"""
BEAST Trading System - Core Engine
Main orchestration engine with strict confidence-based trading
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import uuid

from config.settings import config, CONFIDENCE_THRESHOLD
from core.data_manager import DataManager
from core.decision_maker import DecisionMaker
from analysis.technical import TechnicalAnalyzer
from analysis.blockchain import BlockchainAnalyzer
from analysis.market import MarketAnalyzer
from analysis.ml_predictor import MLPredictor
from analysis.patterns import PatternAnalyzer
from execution.strategy import StrategySelector
from execution.risk_manager import RiskManager
from execution.order_manager import OrderManager
from utils.logger import get_logger, log_trade_decision, log_performance
from utils.metrics import MetricsCollector
from utils.validators import validate_analysis_results

logger = get_logger(__name__)

@dataclass
class TradingSession:
    """Represents a trading session with state tracking"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    trades_executed: int = 0
    trades_attempted: int = 0
    total_profit_loss: float = 0.0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    active_positions: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        if self.trades_attempted == 0:
            return 0.0
        return self.trades_executed / self.trades_attempted

class BeastEngine:
    """
    Main trading engine that orchestrates all components
    NO RANDOM TRADES - Only trades with confidence > 60%
    """
    
    def __init__(self):
        """Initialize the BEAST trading engine"""
        self.config = config
        self.session = TradingSession()
        self.is_running = False
        
        # Initialize components
        self._initialize_components()
        
        # Performance tracking
        self.metrics = MetricsCollector()
        
        # Thread pool for parallel analysis
        self.executor = ThreadPoolExecutor(max_workers=6)
        
        logger.info(f"BEAST Engine initialized - Session: {self.session.session_id}")
        logger.info(f"Confidence threshold: {CONFIDENCE_THRESHOLD * 100}%")
        
    def _initialize_components(self):
        """Initialize all trading components"""
        try:
            # Core components
            self.data_manager = DataManager(self.config)
            self.decision_maker = DecisionMaker(self.config)
            
            # Analysis modules
            self.technical_analyzer = TechnicalAnalyzer(self.config)
            self.blockchain_analyzer = BlockchainAnalyzer(self.config)
            self.market_analyzer = MarketAnalyzer(self.config)
            self.pattern_analyzer = PatternAnalyzer(self.config)
            self.ml_predictor = MLPredictor(self.config)
            
            # Execution components
            self.strategy_selector = StrategySelector(self.config)
            self.risk_manager = RiskManager(self.config)
            self.order_manager = OrderManager(self.config)
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            raise
    
    async def start(self, symbols: Optional[List[str]] = None):
        """Start the trading engine"""
        if self.is_running:
            logger.warning("Engine already running")
            return
            
        self.is_running = True
        symbols = symbols or self.config.trading.enabled_pairs
        
        logger.info(f"Starting BEAST Engine for symbols: {symbols}")
        
        try:
            # Main trading loop
            while self.is_running:
                await self._trading_cycle(symbols)
                
                # Rate limiting
                await asyncio.sleep(self.config.exchange.requests_per_second)
                
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error(f"Engine error: {e}")
            self.session.errors.append({
                'timestamp': datetime.now(timezone.utc),
                'error': str(e),
                'traceback': traceback.format_exc()
            })
        finally:
            await self.stop()
    
    async def _trading_cycle(self, symbols: List[str]):
        """Execute one trading cycle for all symbols"""
        for symbol in symbols:
            try:
                # Check daily trade limit
                if self.session.trades_executed >= self.config.trading.max_daily_trades:
                    logger.info("Daily trade limit reached")
                    break
                
                # Process symbol
                result = await self._process_symbol(symbol)
                
                # Log decision
                log_trade_decision(
                    symbol=symbol,
                    decision=result['decision'],
                    confidence=result['confidence'],
                    reasoning=result['reasoning']
                )
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                self.metrics.record_error('symbol_processing', str(e))
    
    async def _process_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Process a single symbol through the entire pipeline
        Returns decision with NO RANDOM TRADES
        """
        self.session.trades_attempted += 1
        
        result = {
            'symbol': symbol,
            'timestamp': datetime.now(timezone.utc),
            'decision': 'NO_TRADE',  # Default to NO_TRADE
            'confidence': 0.0,
            'reasoning': {},
            'execution': None
        }
        
        try:
            # Step 1: Data Collection & Validation
            logger.debug(f"Collecting data for {symbol}")
            data = await self.data_manager.collect_and_process(symbol)
            
            if not data or data.get('quality_score', 0) < 0.6:
                result['reasoning']['data'] = 'Insufficient data quality'
                return result
            
            # Step 2: Parallel Analysis
            logger.debug(f"Running analysis modules for {symbol}")
            analysis_results = await self._run_analysis(symbol, data)
            
            # Validate analysis results
            if not validate_analysis_results(analysis_results):
                result['reasoning']['analysis'] = 'Invalid analysis results'
                return result
            
            # Step 3: Risk Assessment
            risk_assessment = await self.risk_manager.assess(
                symbol=symbol,
                analysis=analysis_results,
                portfolio=self.session.active_positions
            )
            
            if risk_assessment['risk_score'] > self.config.trading.max_risk_score:
                result['reasoning']['risk'] = f"Risk too high: {risk_assessment['risk_score']:.2f}"
                return result
            
            # Step 4: Decision Making (CRITICAL - 60% threshold)
            decision_result = self.decision_maker.make_decision(
                symbol=symbol,
                analysis_results=analysis_results,
                risk_assessment=risk_assessment,
                data_quality=data['quality_score']
            )
            
            result.update(decision_result)
            
            # Step 5: Execute Trade (only if confidence > 60%)
            if decision_result['decision'] == 'TRADE' and decision_result['confidence'] >= CONFIDENCE_THRESHOLD:
                # Select strategy
                strategy = self.strategy_selector.select(
                    analysis_results=analysis_results,
                    risk_profile=risk_assessment
                )
                
                if not strategy:
                    result['decision'] = 'NO_TRADE'
                    result['reasoning']['strategy'] = 'No valid strategy found'
                    return result
                
                # Execute trade
                execution_result = await self.order_manager.execute(
                    symbol=symbol,
                    strategy=strategy,
                    risk_params=risk_assessment
                )
                
                if execution_result['status'] == 'success':
                    self.session.trades_executed += 1
                    result['execution'] = execution_result
                    
                    # Track position
                    self.session.active_positions[symbol] = {
                        'strategy': strategy,
                        'entry': execution_result['entry_price'],
                        'size': execution_result['size'],
                        'timestamp': datetime.now(timezone.utc)
                    }
                else:
                    result['decision'] = 'NO_TRADE'
                    result['reasoning']['execution'] = execution_result.get('error', 'Execution failed')
            
        except TimeoutError:
            logger.error(f"Timeout processing {symbol}")
            result['reasoning']['error'] = 'Processing timeout'
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            result['reasoning']['error'] = str(e)
        
        # Record metrics
        self.metrics.record_decision(result)
        
        return result
    
    async def _run_analysis(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run all analysis modules in parallel"""
        analysis_tasks = {
            'technical': self._run_with_timeout(
                self.technical_analyzer.analyze(data), 
                timeout=5.0
            ),
            'patterns': self._run_with_timeout(
                self.pattern_analyzer.analyze(data), 
                timeout=5.0
            ),
            'blockchain': self._run_with_timeout(
                self.blockchain_analyzer.analyze(symbol, data), 
                timeout=10.0
            ),
            'market': self._run_with_timeout(
                self.market_analyzer.analyze(symbol, data), 
                timeout=8.0
            ),
            'ml_predictor': self._run_with_timeout(
                self.ml_predictor.predict(data), 
                timeout=5.0
            )
        }
        
        # Execute all analysis tasks in parallel
        results = {}
        for module, task in analysis_tasks.items():
            try:
                results[module] = await task
            except Exception as e:
                logger.warning(f"{module} analysis failed: {e}")
                results[module] = {
                    'status': 'failed',
                    'confidence': 0.0,
                    'error': str(e)
                }
        
        return results
    
    async def _run_with_timeout(self, coro, timeout: float):
        """Run coroutine with timeout"""
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Operation timed out after {timeout}s")
    
    async def stop(self):
        """Stop the trading engine gracefully"""
        logger.info("Stopping BEAST Engine...")
        self.is_running = False
        
        # Close all positions
        if self.session.active_positions:
            logger.info(f"Closing {len(self.session.active_positions)} positions")
            for symbol, position in self.session.active_positions.items():
                try:
                    await self.order_manager.close_position(symbol, position)
                except Exception as e:
                    logger.error(f"Failed to close position for {symbol}: {e}")
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Log session summary
        self._log_session_summary()
        
        # Save session data
        self._save_session_data()
        
        logger.info("BEAST Engine stopped")
    
    def _log_session_summary(self):
        """Log trading session summary"""
        summary = {
            'session_id': self.session.session_id,
            'duration': (datetime.now(timezone.utc) - self.session.start_time).total_seconds() / 3600,
            'trades_attempted': self.session.trades_attempted,
            'trades_executed': self.session.trades_executed,
            'success_rate': self.session.success_rate,
            'total_pnl': self.session.total_profit_loss,
            'errors': len(self.session.errors)
        }
        
        logger.info("=== SESSION SUMMARY ===")
        for key, value in summary.items():
            logger.info(f"{key}: {value}")
        
        # Log to metrics
        log_performance(summary)
    
    def _save_session_data(self):
        """Save session data for analysis"""
        session_file = f"logs/sessions/session_{self.session.session_id}.json"
        
        try:
            session_data = {
                'session': self.session.__dict__,
                'metrics': self.metrics.get_summary(),
                'config': self.config.to_dict()
            }
            
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
                
            logger.info(f"Session data saved to {session_file}")
            
        except Exception as e:
            logger.error(f"Failed to save session data: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current engine status"""
        return {
            'is_running': self.is_running,
            'session_id': self.session.session_id,
            'uptime': (datetime.now(timezone.utc) - self.session.start_time).total_seconds(),
            'trades_executed': self.session.trades_executed,
            'active_positions': len(self.session.active_positions),
            'total_pnl': self.session.total_profit_loss,
            'health': self._check_health()
        }
    
    def _check_health(self) -> Dict[str, bool]:
        """Check health of all components"""
        return {
            'data_manager': self.data_manager.is_healthy(),
            'analyzers': all([
                self.technical_analyzer.is_healthy(),
                self.blockchain_analyzer.is_healthy(),
                self.market_analyzer.is_healthy(),
                self.pattern_analyzer.is_healthy()
            ]),
            'execution': self.order_manager.is_healthy(),
            'risk_manager': self.risk_manager.is_healthy()
        }

# Async entry point
async def main():
    """Main entry point for the BEAST trading system"""
    engine = BeastEngine()
    
    try:
        # Start engine with configured symbols
        await engine.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        await engine.stop()

if __name__ == "__main__":
    # Run the trading engine
    asyncio.run(main())