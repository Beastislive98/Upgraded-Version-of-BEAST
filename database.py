"""
BEAST Trading System - Database Interface
Handles all database operations with support for multiple backends
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import json
import asyncio
import asyncpg
import aiomysql
import aiosqlite
from contextlib import asynccontextmanager
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, String, Float, Integer, DateTime, Boolean, JSON, Index, UniqueConstraint
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, AsyncEngine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import select, insert, update, delete
from sqlalchemy.pool import NullPool

from config.settings import config

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Unified database manager supporting PostgreSQL, MySQL, and SQLite
    """
    
    def __init__(self):
        self.config = config.database
        self.engine: Optional[AsyncEngine] = None
        self.async_session = None
        self.metadata = MetaData()
        
        # Define tables
        self._define_tables()
        
        # Connection pool
        self._pool = None
        
        logger.info(f"DatabaseManager initialized for {self.config.db_type}")
    
    def _define_tables(self):
        """Define database tables"""
        # Trades table
        self.trades_table = Table(
            'trades',
            self.metadata,
            Column('trade_id', String(50), primary_key=True),
            Column('symbol', String(20), nullable=False),
            Column('strategy', String(50), nullable=False),
            Column('side', String(10), nullable=False),
            Column('entry_time', DateTime(timezone=True), nullable=False),
            Column('exit_time', DateTime(timezone=True)),
            Column('entry_price', Float, nullable=False),
            Column('exit_price', Float),
            Column('quantity', Float, nullable=False),
            Column('profit_loss', Float),
            Column('profit_loss_pct', Float),
            Column('commission', Float, default=0),
            Column('is_winner', Boolean),
            Column('metadata', JSON),
            Column('created_at', DateTime(timezone=True), default=datetime.now(timezone.utc)),
            Index('idx_trades_symbol', 'symbol'),
            Index('idx_trades_strategy', 'strategy'),
            Index('idx_trades_entry_time', 'entry_time')
        )
        
        # Decisions table
        self.decisions_table = Table(
            'decisions',
            self.metadata,
            Column('decision_id', String(50), primary_key=True),
            Column('timestamp', DateTime(timezone=True), nullable=False),
            Column('symbol', String(20), nullable=False),
            Column('decision', String(20), nullable=False),
            Column('confidence', Float, nullable=False),
            Column('risk_score', Float),
            Column('reasoning', JSON),
            Column('module_scores', JSON),
            Column('created_at', DateTime(timezone=True), default=datetime.now(timezone.utc)),
            Index('idx_decisions_symbol', 'symbol'),
            Index('idx_decisions_timestamp', 'timestamp'),
            Index('idx_decisions_decision', 'decision')
        )
        
        # Market data table (for caching)
        self.market_data_table = Table(
            'market_data',
            self.metadata,
            Column('id', Integer, primary_key=True, autoincrement=True),
            Column('symbol', String(20), nullable=False),
            Column('timestamp', DateTime(timezone=True), nullable=False),
            Column('open', Float),
            Column('high', Float),
            Column('low', Float),
            Column('close', Float),
            Column('volume', Float),
            Column('timeframe', String(10)),
            Column('created_at', DateTime(timezone=True), default=datetime.now(timezone.utc)),
            UniqueConstraint('symbol', 'timestamp', 'timeframe', name='uq_market_data'),
            Index('idx_market_data_symbol_time', 'symbol', 'timestamp')
        )
        
        # Strategy performance table
        self.strategy_performance_table = Table(
            'strategy_performance',
            self.metadata,
            Column('strategy_name', String(50), primary_key=True),
            Column('total_trades', Integer, default=0),
            Column('winning_trades', Integer, default=0),
            Column('total_profit_loss', Float, default=0),
            Column('win_rate', Float, default=0),
            Column('profit_factor', Float, default=0),
            Column('sharpe_ratio', Float, default=0),
            Column('max_drawdown', Float, default=0),
            Column('last_updated', DateTime(timezone=True)),
            Column('metadata', JSON)
        )
        
        # System metrics table
        self.system_metrics_table = Table(
            'system_metrics',
            self.metadata,
            Column('metric_id', String(50), primary_key=True),
            Column('timestamp', DateTime(timezone=True), nullable=False),
            Column('metric_type', String(50), nullable=False),
            Column('metric_value', Float),
            Column('metadata', JSON),
            Column('created_at', DateTime(timezone=True), default=datetime.now(timezone.utc)),
            Index('idx_metrics_type_time', 'metric_type', 'timestamp')
        )
        
        # Errors table
        self.errors_table = Table(
            'errors',
            self.metadata,
            Column('error_id', String(50), primary_key=True),
            Column('timestamp', DateTime(timezone=True), nullable=False),
            Column('component', String(50), nullable=False),
            Column('error_type', String(100)),
            Column('error_message', String(1000)),
            Column('context', JSON),
            Column('created_at', DateTime(timezone=True), default=datetime.now(timezone.utc)),
            Index('idx_errors_component', 'component'),
            Index('idx_errors_timestamp', 'timestamp')
        )
    
    async def initialize(self):
        """Initialize database connection"""
        try:
            # Create engine based on database type
            if self.config.db_type == 'postgresql':
                self.engine = create_async_engine(
                    self.config.connection_string.replace('postgresql://', 'postgresql+asyncpg://'),
                    pool_size=self.config.pool_size,
                    max_overflow=self.config.max_overflow,
                    pool_timeout=self.config.pool_timeout,
                    echo=False
                )
            elif self.config.db_type == 'mysql':
                self.engine = create_async_engine(
                    self.config.connection_string.replace('mysql://', 'mysql+aiomysql://'),
                    pool_size=self.config.pool_size,
                    max_overflow=self.config.max_overflow,
                    pool_timeout=self.config.pool_timeout,
                    echo=False
                )
            else:  # SQLite
                self.engine = create_async_engine(
                    f"sqlite+aiosqlite:///{self.config.database}.db",
                    echo=False,
                    poolclass=NullPool  # SQLite doesn't support connection pooling
                )
            
            # Create session factory
            self.async_session = sessionmaker(
                self.engine, 
                class_=AsyncSession, 
                expire_on_commit=False
            )
            
            # Create tables if they don't exist
            async with self.engine.begin() as conn:
                await conn.run_sync(self.metadata.create_all)
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    async def close(self):
        """Close database connections"""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connections closed")
    
    @asynccontextmanager
    async def get_session(self):
        """Get database session"""
        async with self.async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    
    # Trade operations
    async def insert_trade(self, trade_data: Dict[str, Any]) -> bool:
        """Insert trade record"""
        try:
            async with self.get_session() as session:
                stmt = insert(self.trades_table).values(**trade_data)
                await session.execute(stmt)
                logger.debug(f"Trade inserted: {trade_data.get('trade_id')}")
                return True
        except Exception as e:
            logger.error(f"Failed to insert trade: {e}")
            return False
    
    async def update_trade(self, trade_id: str, update_data: Dict[str, Any]) -> bool:
        """Update trade record"""
        try:
            async with self.get_session() as session:
                stmt = (
                    update(self.trades_table)
                    .where(self.trades_table.c.trade_id == trade_id)
                    .values(**update_data)
                )
                result = await session.execute(stmt)
                return result.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to update trade: {e}")
            return False
    
    async def get_trade(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """Get trade by ID"""
        try:
            async with self.get_session() as session:
                stmt = select(self.trades_table).where(
                    self.trades_table.c.trade_id == trade_id
                )
                result = await session.execute(stmt)
                row = result.fetchone()
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"Failed to get trade: {e}")
            return None
    
    async def get_trades_by_symbol(
        self, 
        symbol: str, 
        limit: int = 100,
        start_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get trades for a symbol"""
        try:
            async with self.get_session() as session:
                stmt = select(self.trades_table).where(
                    self.trades_table.c.symbol == symbol
                )
                
                if start_date:
                    stmt = stmt.where(self.trades_table.c.entry_time >= start_date)
                
                stmt = stmt.order_by(self.trades_table.c.entry_time.desc()).limit(limit)
                
                result = await session.execute(stmt)
                return [dict(row) for row in result.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get trades: {e}")
            return []
    
    # Decision operations
    async def insert_decision(self, decision_data: Dict[str, Any]) -> bool:
        """Insert decision record"""
        try:
            # Generate decision ID if not provided
            if 'decision_id' not in decision_data:
                decision_data['decision_id'] = f"dec_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            
            async with self.get_session() as session:
                stmt = insert(self.decisions_table).values(**decision_data)
                await session.execute(stmt)
                return True
        except Exception as e:
            logger.error(f"Failed to insert decision: {e}")
            return False
    
    async def get_recent_decisions(
        self, 
        limit: int = 100,
        decision_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get recent decisions"""
        try:
            async with self.get_session() as session:
                stmt = select(self.decisions_table)
                
                if decision_type:
                    stmt = stmt.where(self.decisions_table.c.decision == decision_type)
                
                stmt = stmt.order_by(self.decisions_table.c.timestamp.desc()).limit(limit)
                
                result = await session.execute(stmt)
                return [dict(row) for row in result.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get decisions: {e}")
            return []
    
    # Market data operations
    async def insert_market_data(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> bool:
        """Insert market data (single or batch)"""
        try:
            async with self.get_session() as session:
                if isinstance(data, list):
                    stmt = insert(self.market_data_table)
                    await session.execute(stmt, data)
                else:
                    stmt = insert(self.market_data_table).values(**data)
                    await session.execute(stmt)
                return True
        except Exception as e:
            logger.error(f"Failed to insert market data: {e}")
            return False
    
    async def get_market_data(
        self,
        symbol: str,
        timeframe: str = '1h',
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Get market data as DataFrame"""
        try:
            async with self.get_session() as session:
                stmt = select(self.market_data_table).where(
                    (self.market_data_table.c.symbol == symbol) &
                    (self.market_data_table.c.timeframe == timeframe)
                )
                
                if start_time:
                    stmt = stmt.where(self.market_data_table.c.timestamp >= start_time)
                if end_time:
                    stmt = stmt.where(self.market_data_table.c.timestamp <= end_time)
                
                stmt = stmt.order_by(self.market_data_table.c.timestamp)
                
                if limit:
                    stmt = stmt.limit(limit)
                
                result = await session.execute(stmt)
                data = [dict(row) for row in result.fetchall()]
                
                if data:
                    df = pd.DataFrame(data)
                    df.set_index('timestamp', inplace=True)
                    return df[['open', 'high', 'low', 'close', 'volume']]
                else:
                    return pd.DataFrame()
                    
        except Exception as e:
            logger.error(f"Failed to get market data: {e}")
            return pd.DataFrame()
    
    # Strategy performance operations
    async def update_strategy_performance(
        self, 
        strategy_name: str, 
        performance_data: Dict[str, Any]
    ) -> bool:
        """Update strategy performance metrics"""
        try:
            performance_data['last_updated'] = datetime.now(timezone.utc)
            
            async with self.get_session() as session:
                # Try update first
                stmt = (
                    update(self.strategy_performance_table)
                    .where(self.strategy_performance_table.c.strategy_name == strategy_name)
                    .values(**performance_data)
                )
                result = await session.execute(stmt)
                
                # If no rows updated, insert new record
                if result.rowcount == 0:
                    performance_data['strategy_name'] = strategy_name
                    stmt = insert(self.strategy_performance_table).values(**performance_data)
                    await session.execute(stmt)
                
                return True
        except Exception as e:
            logger.error(f"Failed to update strategy performance: {e}")
            return False
    
    async def get_strategy_performance(self) -> List[Dict[str, Any]]:
        """Get all strategy performance records"""
        try:
            async with self.get_session() as session:
                stmt = select(self.strategy_performance_table).order_by(
                    self.strategy_performance_table.c.total_profit_loss.desc()
                )
                result = await session.execute(stmt)
                return [dict(row) for row in result.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get strategy performance: {e}")
            return []
    
    # System metrics operations
    async def insert_metric(self, metric_data: Dict[str, Any]) -> bool:
        """Insert system metric"""
        try:
            if 'metric_id' not in metric_data:
                metric_data['metric_id'] = f"metric_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            
            async with self.get_session() as session:
                stmt = insert(self.system_metrics_table).values(**metric_data)
                await session.execute(stmt)
                return True
        except Exception as e:
            logger.error(f"Failed to insert metric: {e}")
            return False
    
    async def get_metrics_by_type(
        self,
        metric_type: str,
        hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Get metrics by type for the last N hours"""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            
            async with self.get_session() as session:
                stmt = select(self.system_metrics_table).where(
                    (self.system_metrics_table.c.metric_type == metric_type) &
                    (self.system_metrics_table.c.timestamp >= cutoff_time)
                ).order_by(self.system_metrics_table.c.timestamp)
                
                result = await session.execute(stmt)
                return [dict(row) for row in result.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return []
    
    # Error operations
    async def insert_error(self, error_data: Dict[str, Any]) -> bool:
        """Insert error record"""
        try:
            if 'error_id' not in error_data:
                error_data['error_id'] = f"err_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            
            async with self.get_session() as session:
                stmt = insert(self.errors_table).values(**error_data)
                await session.execute(stmt)
                return True
        except Exception as e:
            logger.error(f"Failed to insert error: {e}")
            return False
    
    async def get_recent_errors(
        self,
        component: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get recent errors"""
        try:
            async with self.get_session() as session:
                stmt = select(self.errors_table)
                
                if component:
                    stmt = stmt.where(self.errors_table.c.component == component)
                
                stmt = stmt.order_by(self.errors_table.c.timestamp.desc()).limit(limit)
                
                result = await session.execute(stmt)
                return [dict(row) for row in result.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get errors: {e}")
            return []
    
    # Analytics queries
    async def get_trade_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get comprehensive trade statistics"""
        try:
            async with self.get_session() as session:
                # Base query
                stmt = select(self.trades_table)
                
                if start_date:
                    stmt = stmt.where(self.trades_table.c.entry_time >= start_date)
                if end_date:
                    stmt = stmt.where(self.trades_table.c.entry_time <= end_date)
                
                result = await session.execute(stmt)
                trades = result.fetchall()
                
                if not trades:
                    return {}
                
                # Calculate statistics
                total_trades = len(trades)
                winning_trades = sum(1 for t in trades if t.is_winner)
                total_pnl = sum(t.profit_loss or 0 for t in trades)
                
                return {
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': total_trades - winning_trades,
                    'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                    'total_pnl': total_pnl,
                    'average_pnl': total_pnl / total_trades if total_trades > 0 else 0,
                    'best_trade': max((t.profit_loss or 0 for t in trades), default=0),
                    'worst_trade': min((t.profit_loss or 0 for t in trades), default=0)
                }
        except Exception as e:
            logger.error(f"Failed to get trade statistics: {e}")
            return {}
    
    async def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old data"""
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
            
            async with self.get_session() as session:
                # Clean market data
                stmt = delete(self.market_data_table).where(
                    self.market_data_table.c.created_at < cutoff_date
                )
                result = await session.execute(stmt)
                logger.info(f"Deleted {result.rowcount} old market data records")
                
                # Clean old metrics
                stmt = delete(self.system_metrics_table).where(
                    self.system_metrics_table.c.created_at < cutoff_date
                )
                result = await session.execute(stmt)
                logger.info(f"Deleted {result.rowcount} old metric records")
                
                # Clean old errors
                stmt = delete(self.errors_table).where(
                    self.errors_table.c.created_at < cutoff_date
                )
                result = await session.execute(stmt)
                logger.info(f"Deleted {result.rowcount} old error records")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
    
    async def backup_database(self, backup_path: str):
        """Create database backup (SQLite only)"""
        if self.config.db_type != 'sqlite':
            logger.warning("Backup only supported for SQLite")
            return False
        
        try:
            import shutil
            source_path = f"{self.config.database}.db"
            shutil.copy2(source_path, backup_path)
            logger.info(f"Database backed up to {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to backup database: {e}")
            return False

# Global database instance
db = DatabaseManager()

# Convenience functions
async def initialize_database():
    """Initialize database"""
    await db.initialize()

async def close_database():
    """Close database"""
    await db.close()

async def insert_trade(trade_data: Dict[str, Any]) -> bool:
    """Insert trade record"""
    return await db.insert_trade(trade_data)

async def insert_decision(decision_data: Dict[str, Any]) -> bool:
    """Insert decision record"""
    return await db.insert_decision(decision_data)

async def get_trade_statistics(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> Dict[str, Any]:
    """Get trade statistics"""
    return await db.get_trade_statistics(start_date, end_date)