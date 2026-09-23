"""Clean Metro_Interstate_Traffic_Volume CSV. Run: python pipeline.py"""

import logging
import sys
from pathlib import Path

import pandas as pd

EXPECTED_COLS = [
    "holiday",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "weather_main",
    "weather_description",
    "date_time",
    "traffic_volume",
]

RAW_PATH = Path("data/metro_interstate_traffic_volume.csv")
CLEAN_PATH = Path("data/cleaned_metro_interstate_traffic_volume.csv")
logger = logging.getLogger(__name__)


def setup_logging():
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler("pipeline.log", mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.handlers.clear()
    root.addHandler(fh)
    root.addHandler(ch)


def load_raw(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        logger.error("CSV not found: %s", path, exc_info=True)
        sys.exit(1)
    except pd.errors.ParserError:
        logger.error("Could not parse CSV: %s", path, exc_info=True)
        sys.exit(1)
    logger.info("Loaded raw data: %s rows, %s columns", df.shape[0], df.shape[1])
    return df


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        logger.error("Missing required columns: %s", missing)
        sys.exit(1)
    logger.info("Schema validation passed")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    before = df["holiday"].astype(str)
    df["holiday"] = before.str.strip().str.title()
    n_case = int((before.str.strip().str.title() != before).sum())
    if n_case:
        logger.warning("Standardised holiday casing for %s rows", n_case)
    else:
        logger.info("holiday already consistent")

    df["weather_main"] = before.str.strip().str.title()
    n_case = int((before.str.strip().str.title() != before).sum())
    if n_case:
        logger.warning("Standardised weather_main casing for %s rows", n_case)
    else:
        logger.info("weather_main already consistent")

    df["weather_description"] = before.str.strip().str.title()
    n_case = int((before.str.strip().str.title() != before).sum())
    if n_case:
        logger.warning("Standardised weather_description casing for %s rows", n_case)
    else:
        logger.info("weather_description already consistent")

    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    bad = int(df["date_time"].isna().sum())
    if bad:
        df = df.dropna(subset=["date_time"]).copy()
        logger.warning("Dropped %s rows with invalid date_time", bad)
    else:
        logger.info("All date_time values parsed OK")

    n_dup = int(df.duplicated().sum())
    if n_dup:
        df = df.drop_duplicates().copy()
        logger.warning("Removed %s duplicate rows", n_dup)
    else:
        logger.info("No duplicate rows found")

    df["month"] = df["date_time"].dt.month

    # sensor error: temp = 0
    bad_t = df["temp"] < 1
    n_t = int(bad_t.sum())
    if n_t:
        for m in df.loc[bad_t, "month"].unique():
            med = df.loc[(df["month"] == m) & (~bad_t), "temp"].median()
            df.loc[bad_t & (df["month"] == m), "temp"] = med
        logger.warning("Imputed %s impossible temp values with monthly median", n_t)
    else:
        logger.info("No impossible temp values found")

    # rain_1h > 1000 is not realistic for 1 hour
    bad_r = df["rain_1h"] > 1000
    n_r = int(bad_r.sum())
    if n_r:
        for m in df.loc[bad_r, "month"].unique():
            med = df.loc[(df["month"] == m) & (~bad_r), "rain_1h"].median()
            df.loc[bad_r & (df["month"] == m), "rain_1h"] = med
        logger.warning("Imputed %s extreme rain_1h values with monthly median", n_r)
    else:
        logger.info("No extreme rain_1h values found")

    df = df.drop(columns=["month"])
    logger.info("Cleaning finished: %s rows, %s columns", df.shape[0], df.shape[1])
    return df


def main():
    setup_logging()
    logger.info("Pipeline started")
    try:
        df = load_raw(RAW_PATH)
        validate_schema(df)
        df = clean_data(df)
        df.to_csv(CLEAN_PATH, index=False)
        logger.info("Saved cleaned data to %s", CLEAN_PATH)
        logger.info("Pipeline completed successfully")
    except Exception:
        logger.error("Pipeline failed", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
