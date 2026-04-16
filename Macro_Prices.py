import argparse
import csv
import json
import logging
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

DB_PATH = Path("data") / "project.db"
EXPORT_DIR = Path("data") / "macro"
LOG_DIR = Path("logs")

DEFAULT_DAYS = 180

YFINANCE_INDICATORS = {
    "CL=F":      {"name": "Crude Oil WTI",     "short": "OIL"},
    "DX-Y.NYB":  {"name": "US Dollar Index",   "short": "DXY"},
    "GC=F":      {"name": "Gold",              "short": "GOLD"},
    "^VIX":      {"name": "VIX Volatility",    "short": "VIX"},
    "^GSPC":     {"name": "S&P 500",           "short": "SP500"},
}
FEAR_GREED_URL = "https://api.alternative.me/fng/"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "macro_collector.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS macro_indicators (
            indicator       TEXT NOT NULL,
            indicator_name  TEXT NOT NULL,
            date            TEXT NOT NULL,
            open            REAL,
            high            REAL,
            low             REAL,
            close           REAL NOT NULL,
            volume          REAL DEFAULT 0.0,
            PRIMARY KEY (indicator, date)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS fear_greed (
            date            TEXT PRIMARY KEY,
            value           INTEGER NOT NULL,
            classification  TEXT NOT NULL
        )
    """)

    c.execute("CREATE INDEX IF NOT EXISTS idx_macro_indicator ON macro_indicators(indicator)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_macro_date ON macro_indicators(date)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_fg_date ON fear_greed(date)")

    conn.commit()
    return conn


def fetch_yfinance_indicator(ticker: str, start_date: str, end_date: str) -> list[dict]:
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. Run: pip install yfinance")
        return []

    info = YFINANCE_INDICATORS[ticker]
    logger.info(f"  Fetching {info['name']} ({ticker})...")

    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval="1d",
            progress=False,
            auto_adjust=True,
        )

        if data.empty:
            logger.warning(f"  No data returned for {ticker}")
            return []

        if hasattr(data.columns, 'levels') and len(data.columns.levels) > 1:
            data.columns = data.columns.droplevel(1)

        rows = []
        for idx, row in data.iterrows():
            date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, 'strftime') else str(idx)[:10]
            rows.append({
                "indicator":      info["short"],
                "indicator_name": info["name"],
                "date":           date_str,
                "open":           float(row["Open"]) if pd.notna(row["Open"]) else None,
                "high":           float(row["High"]) if pd.notna(row["High"]) else None,
                "low":            float(row["Low"]) if pd.notna(row["Low"]) else None,
                "close":          float(row["Close"]) if pd.notna(row["Close"]) else None,
                "volume":         float(row["Volume"]) if pd.notna(row.get("Volume", 0)) else 0.0,
            })

        rows = [r for r in rows if r["close"] is not None]
        return rows

    except Exception as e:
        logger.error(f"  Error fetching {ticker}: {e}")
        return []


def fetch_fear_greed(days: int) -> list[dict]:
    logger.info(f"  Fetching Crypto Fear & Greed Index ({days} days)...")

    try:
        resp = requests.get(
            FEAR_GREED_URL,
            params={"limit": days, "format": "json"},
            timeout=30,
            headers={"User-Agent": "CryptoSentimentResearch/1.0"},
        )
        resp.raise_for_status()
        data = resp.json()

        if "data" not in data:
            logger.warning(f"  Unexpected API response: {str(data)[:200]}")
            return []

        rows = []
        for entry in data["data"]:
            # API returns timestamp as string in seconds
            ts = int(entry["timestamp"])
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)

            rows.append({
                "date":           dt.strftime("%Y-%m-%d"),
                "value":          int(entry["value"]),
                "classification": entry["value_classification"],
            })

        logger.info(f"  Got {len(rows)} Fear & Greed readings")
        return rows

    except requests.exceptions.RequestException as e:
        logger.error(f"  Fear & Greed API error: {e}")
        return []
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        logger.error(f"  Fear & Greed parse error: {e}")
        return []

def store_macro(conn: sqlite3.Connection, rows: list[dict]) -> int:
    """Store macro indicator rows with upsert."""
    if not rows:
        return 0

    c = conn.cursor()
    stored = 0

    for r in rows:
        c.execute("""
            INSERT INTO macro_indicators
                (indicator, indicator_name, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(indicator, date) DO UPDATE SET
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                volume = excluded.volume
        """, (
            r["indicator"], r["indicator_name"], r["date"],
            r["open"], r["high"], r["low"], r["close"], r["volume"],
        ))
        stored += 1

    conn.commit()
    return stored


def store_fear_greed(conn: sqlite3.Connection, rows: list[dict]) -> int:
    if not rows:
        return 0

    c = conn.cursor()
    stored = 0

    for r in rows:
        c.execute("""
            INSERT INTO fear_greed (date, value, classification)
            VALUES (?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                value = excluded.value,
                classification = excluded.classification
        """, (r["date"], r["value"], r["classification"]))
        stored += 1

    conn.commit()
    return stored


def run_collection(days: int = DEFAULT_DAYS) -> dict:
    conn = init_db()

    now = datetime.now(timezone.utc)
    end_date = now.strftime("%Y-%m-%d")
    start_date = (now - timedelta(days=days)).strftime("%Y-%m-%d")

    logger.info(f"{'='*60}")
    logger.info(f"MACRO DATA COLLECTION — {days} days")
    logger.info(f"Period: {start_date} → {end_date}")
    logger.info(f"Indicators: {', '.join(i['short'] for i in YFINANCE_INDICATORS.values())} + Fear&Greed")
    logger.info(f"{'='*60}")

    stats = {}
    total = 0
    for ticker in YFINANCE_INDICATORS:
        info = YFINANCE_INDICATORS[ticker]
        try:
            rows = fetch_yfinance_indicator(ticker, start_date, end_date)
            stored = store_macro(conn, rows)
            stats[info["short"]] = stored
            total += stored
            logger.info(f"  {info['short']}: {stored} daily records stored")
        except Exception as e:
            logger.error(f"  ERROR collecting {info['name']}: {e}")
            stats[info["short"]] = -1

        time.sleep(0.5)
    try:
        fg_rows = fetch_fear_greed(days)
        fg_stored = store_fear_greed(conn, fg_rows)
        stats["FEAR_GREED"] = fg_stored
        total += fg_stored
        logger.info(f"  FEAR_GREED: {fg_stored} daily records stored")
    except Exception as e:
        logger.error(f"  ERROR collecting Fear & Greed: {e}")
        stats["FEAR_GREED"] = -1

    logger.info(f"\n{'='*60}")
    logger.info(f"COLLECTION COMPLETE — {total} total records")
    logger.info(f"{'='*60}")

    conn.close()
    return stats


def export_csvs():
    conn = init_db()
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Macro indicators
    df_macro = pd.read_sql_query(
        "SELECT * FROM macro_indicators ORDER BY indicator, date", conn
    )

    if not df_macro.empty:
        logger.info(f"Exporting {len(df_macro)} macro records...")

        # Per-indicator CSVs
        for indicator in df_macro["indicator"].unique():
            ind_df = df_macro[df_macro["indicator"] == indicator]
            path = EXPORT_DIR / f"{indicator.lower()}_daily.csv"
            ind_df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
            logger.info(f"  {indicator}: {path} ({len(ind_df)} days)")

        # Combined macro CSV
        combined_path = EXPORT_DIR / "all_macro_daily.csv"
        df_macro.to_csv(combined_path, index=False, quoting=csv.QUOTE_MINIMAL)
        logger.info(f"  Combined: {combined_path} ({len(df_macro)} records)")

        # Pivot table: wide format with date as index, indicators as columns
        pivot_df = df_macro.pivot_table(
            index="date", columns="indicator", values="close", aggfunc="first"
        ).reset_index()
        pivot_path = EXPORT_DIR / "macro_pivot_daily.csv"
        pivot_df.to_csv(pivot_path, index=False)
        logger.info(f"  Pivot: {pivot_path} ({len(pivot_df)} days)")
    else:
        logger.warning("No macro indicator data to export!")
    df_fg = pd.read_sql_query(
        "SELECT * FROM fear_greed ORDER BY date", conn
    )

    if not df_fg.empty:
        fg_path = EXPORT_DIR / "fear_greed_daily.csv"
        df_fg.to_csv(fg_path, index=False, quoting=csv.QUOTE_MINIMAL)
        logger.info(f"  Fear & Greed: {fg_path} ({len(df_fg)} days)")
    else:
        logger.warning("No Fear & Greed data to export!")

    conn.close()
    logger.info("CSV export complete!")

def detect_gaps():
    conn = init_db()
    gaps = {}

    logger.info(f"\n{'─'*50}")
    logger.info("MACRO DATA GAP DETECTION")
    logger.info(f"{'─'*50}")

    df = pd.read_sql_query("SELECT date, indicator FROM macro_indicators", conn)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])

        for indicator in df["indicator"].unique():
            ind_dates = set(df[df["indicator"] == indicator]["date"].dt.date)
            all_dates = pd.date_range(df["date"].min(), df["date"].max())
            # Only check weekdays (Mon-Fri) for traditional markets
            weekdays = set(d.date() for d in all_dates if d.weekday() < 5)
            missing = sorted(weekdays - ind_dates)
            if missing:
                gaps[indicator] = [str(d) for d in missing]
                logger.warning(f"  {indicator}: {len(missing)} missing weekdays")
            else:
                logger.info(f"  {indicator}: No gaps (weekdays) — {len(ind_dates)} days")

    df_fg = pd.read_sql_query("SELECT date FROM fear_greed", conn)
    if not df_fg.empty:
        df_fg["date"] = pd.to_datetime(df_fg["date"])
        fg_dates = set(df_fg["date"].dt.date)
        all_dates = pd.date_range(df_fg["date"].min(), df_fg["date"].max())
        expected = set(d.date() for d in all_dates)
        missing = sorted(expected - fg_dates)
        if missing:
            gaps["FEAR_GREED"] = [str(d) for d in missing]
            logger.warning(f"  FEAR_GREED: {len(missing)} missing days")
        else:
            logger.info(f"  FEAR_GREED: No gaps — {len(fg_dates)} days")

    conn.close()
    return gaps

def print_status():
    """Print macro data collection statistics."""
    if not DB_PATH.exists():
        print("No database found. Run collection first.")
        return

    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    # Check tables exist
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='macro_indicators'")
    has_macro = c.fetchone() is not None
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='fear_greed'")
    has_fg = c.fetchone() is not None

    if not has_macro and not has_fg:
        print("No macro data collected yet.")
        conn.close()
        return

    print(f"\n{'='*60}")
    print("MACRO DATA STATUS")
    print(f"{'='*60}")

    if has_macro:
        c.execute("SELECT COUNT(*) FROM macro_indicators")
        total = c.fetchone()[0]
        print(f"\nMacro records: {total:,}")

        print(f"\n{'─'*65}")
        print(f"{'Indicator':<10} {'Name':<20} {'Days':>6} {'Start':<12} {'End':<12} {'Last':>10}")
        print(f"{'─'*65}")

        for ticker_info in YFINANCE_INDICATORS.values():
            short = ticker_info["short"]
            c.execute("""
                SELECT COUNT(*), MIN(date), MAX(date)
                FROM macro_indicators WHERE indicator = ?
            """, (short,))
            row = c.fetchone()
            if row and row[0] > 0:
                count, d_min, d_max = row
                c.execute("""
                    SELECT close FROM macro_indicators
                    WHERE indicator = ? ORDER BY date DESC LIMIT 1
                """, (short,))
                last = c.fetchone()[0]
                print(f"{short:<10} {ticker_info['name']:<20} {count:>6} {d_min:<12} {d_max:<12} {last:>10,.2f}")

    if has_fg:
        c.execute("SELECT COUNT(*) FROM fear_greed")
        fg_total = c.fetchone()[0]
        c.execute("SELECT MIN(date), MAX(date) FROM fear_greed")
        fg_min, fg_max = c.fetchone()
        c.execute("SELECT value, classification FROM fear_greed ORDER BY date DESC LIMIT 1")
        fg_last = c.fetchone()

        print(f"\n{'─'*50}")
        print(f"Fear & Greed Index")
        print(f"  Records:    {fg_total}")
        print(f"  Date range: {fg_min} → {fg_max}")
        if fg_last:
            print(f"  Latest:     {fg_last[0]} ({fg_last[1]})")

        # Distribution
        c.execute("""
            SELECT classification, COUNT(*), ROUND(AVG(value), 1)
            FROM fear_greed GROUP BY classification ORDER BY AVG(value)
        """)
        print(f"\n  Distribution:")
        for cls, cnt, avg in c.fetchall():
            print(f"    {cls:<20} {cnt:>4} days  (avg: {avg})")

    conn.close()
    detect_gaps()


def main():
    parser = argparse.ArgumentParser(
        description="Macro Economic Data Collector (yfinance + Fear & Greed API)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python macro_collector.py                    # 180 days
  python macro_collector.py --days 30          # Last 30 days
  python macro_collector.py --export-only      # Export CSVs only
  python macro_collector.py --status           # Show stats
        """,
    )
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS,
                        help=f"Days to collect (default: {DEFAULT_DAYS})")
    parser.add_argument("--export-only", action="store_true",
                        help="Skip collection, just export CSVs")
    parser.add_argument("--status", action="store_true",
                        help="Show collection statistics")

    args = parser.parse_args()

    if args.status:
        print_status()
        return

    if args.export_only:
        export_csvs()
        detect_gaps()
        return

    run_collection(days=args.days)
    export_csvs()
    detect_gaps()


if __name__ == "__main__":
    main()