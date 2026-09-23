"""
CLI for FreshBasket checkout data.

  python app.py summary
  python app.py by-hour --hour 11
  python app.py recommend --day weekend
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

FEAT_PATH = Path("data/features_checkouts.csv")
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
        sys.exit(1)


def cmd_summary(df: pd.DataFrame) -> None:
    print("Rows:", len(df))
    print("Date range:", df["date_time"].min(), "to", df["date_time"].max())
    print("Avg checkouts:", round(df["checkouts"].mean(), 1))
    print("Avg staff:", round(df["staff_count"].mean(), 1))
    print(df["demand_category"].value_counts().to_string())


def cmd_by_hour(df: pd.DataFrame, hour: int) -> None:
    if hour < 0 or hour > 23:
        logger.error("Invalid hour: %s", hour)
        print("Error: hour must be between 0 and 23")
        return
    sub = df[df["hour"] == hour]
    if sub.empty:
        print("No rows for hour", hour)
        return
    print(f"Hour {hour}: {len(sub)} records")
    print("Avg checkouts:", round(sub["checkouts"].mean(), 1))
    print("Top department:", sub["department"].mode().iloc[0])


def cmd_recommend(df: pd.DataFrame, day: str) -> None:
    day = day.lower().strip()
    if day not in ("weekday", "weekend"):
        logger.error("Invalid day type: %s", day)
        print("Error: day must be 'weekday' or 'weekend'")
        return
    sub = df[df["is_weekend"] == (0 if day == "weekday" else 1)]
    by_hour = sub.groupby("hour")["checkouts"].mean().sort_values(ascending=False)
    top = by_hour.head(3)
    print(f"Busiest hours for {day} staffing:")
    for h, v in top.items():
        print(f"  {int(h):02d}:00  avg checkouts {v:.0f}")
    h0 = int(top.index[0])
    print(f"Suggestion: roster more cashiers around {h0:02d}:00 on a {day}.")


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="FreshBasket checkout CLI")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("summary")
    p_h = sub.add_parser("by-hour")
    p_h.add_argument("--hour", type=int, required=True)
    p_r = sub.add_parser("recommend")
    p_r.add_argument("--day", type=str, required=True)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    logger.info("CLI command=%s args=%s", args.command, vars(args))
    df = load_data()
    if args.command == "summary":
        cmd_summary(df)
    elif args.command == "by-hour":
        cmd_by_hour(df, args.hour)
    elif args.command == "recommend":
        cmd_recommend(df, args.day)


if __name__ == "__main__":
    main()
