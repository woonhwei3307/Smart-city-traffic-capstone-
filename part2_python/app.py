"""
CLI for Metro Interstate Traffic Analytics.

Example Usages:
  python app.py query-time --datetime "2012-10-02 09:00:00"
  python app.py high-traffic --congestion Severe
  python app.py compare-days
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

FEAT_PATH = Path("data/features_metro_interstate_traffic_volume.csv")
logger = logging.getLogger(__name__)

def setup_logging():
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler("pipeline.log", mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(ch)

def load_data() -> pd.DataFrame:
    try:
        df = pd.read_csv(FEAT_PATH)
        df["date_time"] = pd.to_datetime(df["date_time"])
        return df
    except FileNotFoundError:
        logger.error("Features file not found: %s", FEAT_PATH, exc_info=True)
        print(f"Error: Processed features file not found at {FEAT_PATH}. Please run your pipeline first.")
        sys.exit(1)

def cmd_query_time(df: pd.DataFrame, datetime_str: str) -> None:
    """Query traffic conditions for a specific date and timestamp."""
    try:
        target_dt = pd.to_datetime(datetime_str)
    except ValueError:
        logger.error("Invalid datetime format supplied: %s", datetime_str)
        print(f"Error: '{datetime_str}' is not a valid date format. Use 'YYYY-MM-DD HH:MM:SS'.")
        return

    sub = df[df["date_time"] == target_dt]
    if sub.empty:
        print(f"No traffic record found for exact timestamp: {target_dt}")
        return

    row = sub.iloc[0]
    print(f"\nTraffic Record for {target_dt}:")
    print(f"  Volume: {row['traffic_volume']} vehicles/hr")
    print(f"  Congestion Category: {row.get('congestion_category', 'N/A')}")
    print(f"  Weather Main: {row['weather_main']} ({row['weather_description']})")
    print(f"  Temperature: {row['temp']:.2f} K")

def cmd_high_traffic(df: pd.DataFrame, congestion_level: str) -> None:
    """Identify periods matching a specific high-traffic congestion level (e.g., High, Severe)."""
    congestion_level = congestion_level.strip().capitalize()
    valid_levels = ["Low", "Medium", "High", "Severe"]

    if congestion_level not in valid_levels:
        logger.error("Invalid congestion level: %s", congestion_level)
        print(f"Error: congestion level must be one of {valid_levels}")
        return

    sub = df[df["congestion_category"] == congestion_level]
    print(f"\nFound {len(sub)} records classified under '{congestion_level}' congestion:")
    
    if not sub.empty:
        sample_peaks = sub.sort_values(by="traffic_volume", ascending=False).head(5)
        print("\nTop 5 peak instances within this category:")
        for _, row in sample_peaks.iterrows():
            print(f"  Time: {row['date_time']} | Volume: {row['traffic_volume']} veh/h | Weather: {row['weather_main']}")

def cmd_compare_days(df: pd.DataFrame) -> None:
    """Compare traffic statistics between weekdays and weekends."""
    summary = df.groupby("is_weekend")["traffic_volume"].agg(["count", "mean", "median", "std"])
    summary = summary.rename(index={0: "Weekday (Mon-Fri)", 1: "Weekend (Sat-Sun)"})
    
    print("\nWeekday vs Weekend Traffic Volume Comparison:")
    print("=" * 60)
    print(summary.to_string())
    print("=" * 60)

    # Find peak hours for each
    for val, label in [(0, "Weekday"), (1, "Weekend")]:
        subset = df[df["is_weekend"] == val]
        busiest_hour = subset.groupby("hour")["traffic_volume"].mean().idxmax()
        print(f"Busiest average hour on {label}s: {int(busiest_hour):02d}:00")

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Metro Interstate Traffic Analytics CLI")
    sub = parser.add_subparsers(dest="command")

    # 1. Query Specific Date/Time Subcommand
    p_time = sub.add_parser("query-time", help="Query traffic for a specific date/time")
    p_time.add_argument("--datetime", type=str, required=True, help="Format: 'YYYY-MM-DD HH:MM:SS'")

    # 2. Identify High-Traffic Periods Subcommand
    p_high = sub.add_parser("high-traffic", help="Filter and examine periods by congestion category")
    p_high.add_argument("--congestion", type=str, default="Peak", help="Category: Low, Medium, High, or Peak")

    # 3. Compare Weekday and Weekend Subcommand
    sub.add_parser("compare-days", help="Compare overall volume and patterns between weekdays and weekends")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    logger.info("CLI command=%s args=%s", args.command, vars(args))
    df = load_data()

    if args.command == "query-time":
        cmd_query_time(df, args.datetime)
    elif args.command == "high-traffic":
        cmd_high_traffic(df, args.congestion)
    elif args.command == "compare-days":
        cmd_compare_days(df)

if __name__ == "__main__":
    main()