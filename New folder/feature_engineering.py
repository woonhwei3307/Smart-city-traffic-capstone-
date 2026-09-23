"""Add features for FreshBasket data. Run: python feature_engineering.py"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CLEAN_PATH = Path("data/cleaned_checkouts.csv")
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


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(
        "Feature engineering start: %s rows, %s columns", df.shape[0], df.shape[1]
    )
    df = df.copy()
    df["date_time"] = pd.to_datetime(df["date_time"])

    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["is_promo"] = df["promo"].notna().astype(int)

    dept_map = {d: i for i, d in enumerate(sorted(df["department"].dropna().unique()))}
    df["department_code"] = df["department"].map(dept_map)
    logger.debug("dept_map=%s", dept_map)

    df["is_fresh_area"] = df["department"].isin(["Produce", "Dairy", "Bakery"]).astype(int)
    df["is_wet_weather"] = (df["rain_mm"] > 0).astype(int)

    for col in ["temp_c", "checkouts"]:
        mn, mx = df[col].min(), df[col].max()
        df[col + "_scaled"] = (df[col] - mn) / (mx - mn) if mx != mn else 0.0
        logger.debug("%s min=%.4f max=%.4f", col, mn, mx)

    q1, q2, q3 = df["checkouts"].quantile([0.25, 0.5, 0.75]).values
    logger.debug("demand quartiles q1=%.1f q2=%.1f q3=%.1f", q1, q2, q3)

    def demand_label(v):
        if v <= q1:
            return "Low"
        if v <= q2:
            return "Medium"
        if v <= q3:
            return "High"
        return "Peak"

    labels = []
    for v in df["checkouts"]:
        labels.append(demand_label(v))
    df["demand_category"] = labels

    logger.info(
        "Feature engineering end: %s rows, %s columns", df.shape[0], df.shape[1]
    )
    return df


def main():
    setup_logging()
    try:
        df = pd.read_csv(CLEAN_PATH)
        logger.info("Loaded cleaned data from %s", CLEAN_PATH)
        df = add_features(df)
        df.to_csv(FEAT_PATH, index=False)
        logger.info("Saved features to %s", FEAT_PATH)
    except FileNotFoundError:
        logger.error("Cleaned file missing. Run pipeline.py first.", exc_info=True)
        sys.exit(1)
    except Exception:
        logger.error("Feature engineering failed", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
