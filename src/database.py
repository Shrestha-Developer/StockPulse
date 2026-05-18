import sqlite3
import pandas as pd
import json
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
from datetime import datetime


# ----------------------------------------
# DATABASE CONNECTION MANAGER
# ----------------------------------------

class DatabaseManager:
    """
    Manages SQLite database connections and operations for the Stock Market Analyzer.
    Provides a centralized interface for all database operations.
    """

    def __init__(self, db_path: str = "db/market.db"):
        """
        Initialize the database manager.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize the database with schema if it doesn't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if tables exist
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='symbols'"
            )
            
            if not cursor.fetchone():
                self._create_schema(cursor)
                conn.commit()
                print(f"Database initialized at {self.db_path}")
            
            conn.close()
        except Exception as e:
            print(f"Error initializing database: {e}")
            raise

    def _create_schema(self, cursor):
        """Create the database schema."""
        schema_sql = """
        CREATE TABLE IF NOT EXISTS symbols(
            ticker TEXT PRIMARY KEY,
            name TEXT,
            exchange TEXT,
            currency TEXT
        );

        CREATE TABLE IF NOT EXISTS candles_daily(
            ticker TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            adj_close REAL,
            volume INTEGER,
            PRIMARY KEY (ticker, date)
        );

        CREATE TABLE IF NOT EXISTS indicators_daily(
            ticker TEXT,
            date TEXT,
            sma20 REAL,
            sma50 REAL,
            rsi14 REAL,
            macd REAL,
            macd_signal REAL,
            macd_hist REAL,
            bb_upper REAL,
            bb_mid REAL,
            bb_lower REAL,
            PRIMARY KEY (ticker, date)
        );

        CREATE TABLE IF NOT EXISTS backtests(
            id TEXT PRIMARY KEY,
            name TEXT,
            params_json TEXT,
            start TEXT,
            end TEXT,
            ticker TEXT,
            pnl REAL,
            max_dd REAL,
            sharpe REAL,
            trades INTEGER,
            win_rate REAL,
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS portfolio_tx(
            id TEXT PRIMARY KEY,
            ticker TEXT,
            ts TEXT,
            side TEXT,
            qty REAL,
            price REAL,
            fees REAL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS alerts(
            id TEXT PRIMARY KEY,
            ticker TEXT,
            rule TEXT,
            threshold REAL,
            active INTEGER DEFAULT 1,
            last_fired TEXT
        );
        """
        
        cursor.executescript(schema_sql)

    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.
        
        Usage:
            with db.get_connection() as conn:
                # use connection
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # ----------------------------------------
    # SYMBOLS TABLE OPERATIONS
    # ----------------------------------------

    def add_symbol(self, ticker: str, name: str, exchange: str, currency: str = "USD"):
        """Add or update a symbol."""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO symbols (ticker, name, exchange, currency)
                VALUES (?, ?, ?, ?)
                """,
                (ticker, name, exchange, currency)
            )
            conn.commit()

    def get_symbol(self, ticker: str) -> Optional[Dict]:
        """Get symbol information."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM symbols WHERE ticker = ?",
                (ticker,)
            ).fetchone()
            return dict(row) if row else None

    def get_all_symbols(self) -> List[Dict]:
        """Get all symbols in database."""
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM symbols").fetchall()
            return [dict(row) for row in rows]

    # ----------------------------------------
    # CANDLES TABLE OPERATIONS
    # ----------------------------------------

    def save_candle(
        self,
        ticker: str,
        date: str,
        open_: float,
        high: float,
        low: float,
        close: float,
        adj_close: float,
        volume: int
    ):
        """Save a single candle (OHLCV) record."""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO candles_daily
                (ticker, date, open, high, low, close, adj_close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (ticker, date, open_, high, low, close, adj_close, volume)
            )
            conn.commit()

    def save_candles_batch(self, ticker: str, df: pd.DataFrame, db_path: Optional[str] = None):
        """
        Save multiple candles from a DataFrame.
        
        Args:
            ticker: Stock ticker symbol
            df: DataFrame with columns: date, open, high, low, close, adj_close, volume
        """
        if df.empty:
            print("No data to save.")
            return

        with self.get_connection() as conn:
            cursor = conn.cursor()
            for row in df.itertuples(index=False):
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO candles_daily
                    (ticker, date, open, high, low, close, adj_close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ticker,
                        row.date,
                        row.open,
                        row.high,
                        row.low,
                        row.close,
                        getattr(row, "adj_close", row.close),
                        row.volume
                    )
                )
            conn.commit()
        print(f"Saved {len(df)} candles for {ticker}")

    def get_candles(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get candle data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)
        
        Returns:
            DataFrame with OHLCV data
        """
        with self.get_connection() as conn:
            query = "SELECT * FROM candles_daily WHERE ticker = ?"
            params = [ticker]
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
            
            query += " ORDER BY date"
            
            df = pd.read_sql_query(query, conn, params=params)
            return df

    # ----------------------------------------
    # INDICATORS TABLE OPERATIONS
    # ----------------------------------------

    def save_indicator(
        self,
        ticker: str,
        date: str,
        sma20: Optional[float] = None,
        sma50: Optional[float] = None,
        rsi14: Optional[float] = None,
        macd: Optional[float] = None,
        macd_signal: Optional[float] = None,
        macd_hist: Optional[float] = None,
        bb_upper: Optional[float] = None,
        bb_mid: Optional[float] = None,
        bb_lower: Optional[float] = None
    ):
        """Save indicator values for a specific date."""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO indicators_daily
                (ticker, date, sma20, sma50, rsi14, macd, macd_signal, macd_hist,
                 bb_upper, bb_mid, bb_lower)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (ticker, date, sma20, sma50, rsi14, macd, macd_signal, macd_hist,
                 bb_upper, bb_mid, bb_lower)
            )
            conn.commit()

    def save_indicators_batch(self, ticker: str, df: pd.DataFrame):
        """Save multiple indicator records from a DataFrame."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for row in df.itertuples(index=False):
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO indicators_daily
                    (ticker, date, sma20, sma50, rsi14, macd, macd_signal, macd_hist,
                     bb_upper, bb_mid, bb_lower)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ticker,
                        row.date,
                        getattr(row, "sma20", None),
                        getattr(row, "sma50", None),
                        getattr(row, "rsi14", None),
                        getattr(row, "macd", None),
                        getattr(row, "macd_signal", None),
                        getattr(row, "macd_hist", None),
                        getattr(row, "bb_upper", None),
                        getattr(row, "bb_mid", None),
                        getattr(row, "bb_lower", None)
                    )
                )
            conn.commit()
        print(f"Saved {len(df)} indicators for {ticker}")

    def get_indicators(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Get indicator data for a ticker."""
        with self.get_connection() as conn:
            query = "SELECT * FROM indicators_daily WHERE ticker = ?"
            params = [ticker]
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
            
            query += " ORDER BY date"
            
            df = pd.read_sql_query(query, conn, params=params)
            return df

    # ----------------------------------------
    # BACKTESTS TABLE OPERATIONS
    # ----------------------------------------

    def save_backtest(
        self,
        id_: str,
        name: str,
        params: Dict[str, Any],
        start: str,
        end: str,
        ticker: str,
        pnl: float,
        max_dd: float,
        sharpe: float,
        trades: int,
        win_rate: float
    ):
        """Save backtest results."""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO backtests
                (id, name, params_json, start, end, ticker, pnl, max_dd, sharpe,
                 trades, win_rate, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    id_,
                    name,
                    json.dumps(params),
                    start,
                    end,
                    ticker,
                    pnl,
                    max_dd,
                    sharpe,
                    trades,
                    win_rate,
                    datetime.now().isoformat()
                )
            )
            conn.commit()

    def get_backtest(self, id_: str) -> Optional[Dict]:
        """Get a specific backtest."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM backtests WHERE id = ?",
                (id_,)
            ).fetchone()
            if row:
                result = dict(row)
                result["params"] = json.loads(result["params_json"])
                return result
            return None

    def get_backtests_for_ticker(self, ticker: str) -> List[Dict]:
        """Get all backtests for a ticker."""
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM backtests WHERE ticker = ? ORDER BY created_at DESC",
                (ticker,)
            ).fetchall()
            results = []
            for row in rows:
                result = dict(row)
                result["params"] = json.loads(result["params_json"])
                results.append(result)
            return results

    # ----------------------------------------
    # PORTFOLIO TRANSACTIONS TABLE OPERATIONS
    # ----------------------------------------

    def add_transaction(
        self,
        id_: str,
        ticker: str,
        ts: str,
        side: str,
        qty: float,
        price: float,
        fees: float = 0
    ):
        """Add a portfolio transaction."""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO portfolio_tx (id, ticker, ts, side, qty, price, fees)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (id_, ticker, ts, side, qty, price, fees)
            )
            conn.commit()

    def get_transactions(self, ticker: Optional[str] = None) -> List[Dict]:
        """Get portfolio transactions."""
        with self.get_connection() as conn:
            if ticker:
                rows = conn.execute(
                    "SELECT * FROM portfolio_tx WHERE ticker = ? ORDER BY ts DESC",
                    (ticker,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM portfolio_tx ORDER BY ts DESC"
                ).fetchall()
            return [dict(row) for row in rows]

    # ----------------------------------------
    # ALERTS TABLE OPERATIONS
    # ----------------------------------------

    def add_alert(
        self,
        id_: str,
        ticker: str,
        rule: str,
        threshold: float,
        active: int = 1
    ):
        """Add an alert rule."""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO alerts (id, ticker, rule, threshold, active)
                VALUES (?, ?, ?, ?, ?)
                """,
                (id_, ticker, rule, threshold, active)
            )
            conn.commit()

    def get_alerts(self, ticker: Optional[str] = None, active_only: bool = True) -> List[Dict]:
        """Get alerts."""
        with self.get_connection() as conn:
            query = "SELECT * FROM alerts"
            params = []
            
            if ticker:
                query += " WHERE ticker = ?"
                params.append(ticker)
            
            if active_only:
                if params:
                    query += " AND active = 1"
                else:
                    query += " WHERE active = 1"
            
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def update_alert_fired(self, id_: str):
        """Update the last_fired timestamp for an alert."""
        with self.get_connection() as conn:
            conn.execute(
                "UPDATE alerts SET last_fired = ? WHERE id = ?",
                (datetime.now().isoformat(), id_)
            )
            conn.commit()

    # ----------------------------------------
    # UTILITY METHODS
    # ----------------------------------------

    def execute_query(self, query: str, params: tuple = ()) -> pd.DataFrame:
        """Execute a custom SQL query and return results as DataFrame."""
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def execute_update(self, query: str, params: tuple = ()):
        """Execute a custom SQL update/insert/delete query."""
        with self.get_connection() as conn:
            conn.execute(query, params)
            conn.commit()

    def get_latest_date(self, ticker: str) -> Optional[str]:
        """Get the latest date available for a ticker."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT MAX(date) as latest_date FROM candles_daily WHERE ticker = ?",
                (ticker,)
            ).fetchone()
            return row["latest_date"] if row and row["latest_date"] else None

    def clear_ticker_data(self, ticker: str):
        """Delete all data for a ticker (use with caution)."""
        with self.get_connection() as conn:
            conn.execute("DELETE FROM candles_daily WHERE ticker = ?", (ticker,))
            conn.execute("DELETE FROM indicators_daily WHERE ticker = ?", (ticker,))
            conn.commit()

    def get_database_stats(self) -> Dict[str, int]:
        """Get statistics about the database."""
        with self.get_connection() as conn:
            stats = {}
            tables = [
                "symbols",
                "candles_daily",
                "indicators_daily",
                "backtests",
                "portfolio_tx",
                "alerts"
            ]
            for table in tables:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                stats[table] = count
            return stats


# ----------------------------------------
# SINGLETON INSTANCE
# ----------------------------------------

_db_instance = None


def get_db(db_path: str = "db/market.db") -> DatabaseManager:
    """Get or create a singleton database manager instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager(db_path)
    return _db_instance


# ----------------------------------------
# MAIN (for testing)
# ----------------------------------------

if __name__ == "__main__":
    db = get_db()
    
    # Test database
    print("\n" + "="*50)
    print("Database Statistics")
    print("="*50)
    stats = db.get_database_stats()
    for table, count in stats.items():
        print(f"{table}: {count} records")
