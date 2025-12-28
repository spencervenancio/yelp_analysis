from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
from datetime import datetime, timedelta
import pandas as pd

from src.config import PROCESSED_DATA_DIR

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold


app = typer.Typer()


def add_total_hours(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """Adds a column to the DataFrame with total hours open per week. Requires 'hours' column to be present."""

    def _process_hours(hours_str) -> pd.Timedelta:
        if hours_str is None or pd.isna(hours_str):
            return pd.Timedelta(0)

        if isinstance(hours_str, (bytes, bytearray)):
            try:
                hours_str = hours_str.decode("utf-8")
            except Exception:
                return pd.Timedelta(0)

        if hours_str is None or pd.isna(hours_str) or not hasattr(hours_str, "values"):
            return pd.Timedelta(0)

        total = pd.Timedelta(0)
        for day in hours_str.values():
            start_str, end_str = day.split("-", 1)
            delta = pd.Timedelta(
                datetime.strptime(end_str, "%H:%M") - datetime.strptime(start_str, "%H:%M")
            )
            if delta < pd.Timedelta(0):
                delta += pd.Timedelta(days=1)
            total += delta

        return total

    if not inplace:
        df = df.copy()

    hours = df.hours.apply(_process_hours)
    df["total_hours_open"] = hours / pd.Timedelta(hours=1)  # convert to hours
    return df


def is_restaurant(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """ Adds a boolean column 'is_restaurant' to indicate if the business is a restaurant."""
    if not inplace:
        df = df.copy()  
    
    return df[df.Restaurants == True]



@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
