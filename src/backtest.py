import os
import argparse
import uuid
from datetime import datetime

import pandas as pd
import numpy as np

try:
    # When imported as a package (e.g. dashboard imports src.backtest)
    from src.database import get_db
except Exception:
    # When executed as a script: python src/backtest.py
    from database import get_db


# ----------------------------------------
# RUN BACKTEST (Project-integrated)
# ----------------------------------------


def run_backtest(
    ticker: str = "AAPL",
    db_path: str = "db/market.db",
    fee_bps: float = 5.0,
    start_date: str = None,
    end_date: str = None,
    save_to_db: bool = True,
    strategy_name: str = "SMA Crossover"
) -> pd.DataFrame:

    print(f"\nRunning backtest for {ticker}...\n")

    db = get_db(db_path)

    # Load candles + indicators
    query = """
        SELECT
            c.date,
            c.close,
            i.sma20,
            i.sma50
        FROM candles_daily c
        JOIN indicators_daily i
          ON c.ticker = i.ticker
         AND c.date = i.date
        WHERE c.ticker = ?
    """

    params = [ticker]

    if start_date:
        query += " AND c.date >= ?"
        params.append(start_date)

    if end_date:
        query += " AND c.date <= ?"
        params.append(end_date)

    query += " ORDER BY c.date"

    df = db.execute_query(query, tuple(params))

    if df.empty:
        print("No data found for ticker. Make sure candles and indicators are populated.")
        return df

    df["date"] = pd.to_datetime(df["date"]).dt.date

    # Signals: simple SMA crossover
    df["signal"] = np.where(df["sma20"] > df["sma50"], 1, 0)
    df["position"] = df["signal"].shift(1).fillna(0)

    # Returns
    df["market_return"] = df["close"].pct_change().fillna(0)
    df["strategy_return"] = df["position"] * df["market_return"]

    # Transaction costs
    df["trade"] = df["position"].diff().abs().fillna(0)
    transaction_cost = fee_bps / 10000.0
    df["strategy_return"] = df["strategy_return"] - (df["trade"] * transaction_cost)

    # Equity curve
    df["equity_curve"] = (1 + df["strategy_return"]).cumprod()

    # Metrics
    total_return = (df["equity_curve"].iloc[-1] - 1) * 100
    sharpe = np.sqrt(252) * (df["strategy_return"].mean() / (df["strategy_return"].std() if df["strategy_return"].std() != 0 else np.nan))
    rolling_max = df["equity_curve"].cummax()
    drawdown = df["equity_curve"] / rolling_max - 1
    max_drawdown = drawdown.min() * 100
    trades = int(df["trade"].sum())
    trade_count_days = (df["trade"] > 0).sum()
    win_rate = (df[ df["strategy_return"] > 0 ].shape[0] / df.shape[0]) * 100

    # Ensure outputs directory
    os.makedirs("outputs", exist_ok=True)

    # Save results
    results = pd.DataFrame({
        "Metric": [
            "Strategy Returns (%)",
            "Sharpe Ratio",
            "Win Rate (%)",
            "Maximum Drawdown (%)",
            "Total Trades",
            "Trade Days"
        ],
        "Value": [
            total_return,
            sharpe,
            win_rate,
            max_drawdown,
            trades,
            trade_count_days
        ]
    })

    results.to_csv("outputs/backtest_results.csv", index=False)
    df.to_csv("outputs/backtest_equity_curve.csv", index=False)

    # Print key outputs
    print("BACKTEST RESULTS")
    print("-" * 40)
    print(f"Strategy Returns   : {total_return:.2f}%")
    print(f"Equity Curve end  : {df['equity_curve'].iloc[-1]:.4f}")
    print(f"Sharpe Ratio      : {sharpe:.2f}")
    print(f"Win Rate          : {win_rate:.2f}%")
    print(f"Maximum Drawdown  : {max_drawdown:.2f}%")
    print(f"Total Trades      : {trades} (on {trade_count_days} days)")

    # Save summary to database backtests table
    if save_to_db:
        bt_id = str(uuid.uuid4())
        start = str(df["date"].iloc[0])
        end = str(df["date"].iloc[-1])
        params = {
            "strategy": strategy_name,
            "fee_bps": fee_bps,
            "sma_short": 20,
            "sma_long": 50
        }
        try:
            db.save_backtest(
                id_=bt_id,
                name=strategy_name,
                params=params,
                start=start,
                end=end,
                ticker=ticker,
                pnl=total_return,
                max_dd=max_drawdown,
                sharpe=sharpe,
                trades=trades,
                win_rate=win_rate
            )
            print(f"Saved backtest {bt_id} to database")
        except Exception as e:
            print(f"Failed to save backtest to database: {e}")

    print("\nBacktest completed successfully! Outputs in outputs/")

    return df


def _parse_args():
    p = argparse.ArgumentParser(description="Run SMA crossover backtest")
    p.add_argument("ticker", nargs="?", default="AAPL")
    p.add_argument("--db", default="db/market.db")
    p.add_argument("--fee-bps", type=float, default=5.0)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--no-save", dest="save", action="store_false", help="Don't save results to DB")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_backtest(
        ticker=args.ticker,
        db_path=args.db,
        fee_bps=args.fee_bps,
        start_date=args.start,
        end_date=args.end,
        save_to_db=args.save
    )