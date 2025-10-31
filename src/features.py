from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
from datetime import datetime, timedelta
import pandas as pd

from src.config import PROCESSED_DATA_DIR

app = typer.Typer()

def add_total_hours(df: pd.DataFrame) -> pd.DataFrame: 
    """ Adds a column to the DataFrame with total hours open per week. Requires 'hours' column to be present."""
    
    def process_hours(hours_str: str) -> timedelta:
        """Processes a string representing opening hours and returns the total open time as a timedelta."""
        if hours_str is None:
            return 0  # return zero time if hours are not available
        total = timedelta()
        for day in hours_str.values():
            start_str, end_str = day.split('-')

            start = datetime.strptime(start_str, "%H:%M")
            end = datetime.strptime(end_str, "%H:%M")

            delta = end - start
            total += delta  # add up timedelta objects
        return total.seconds / 3600  # convert to hour
    
    hours = df.hours.apply(process_hours)
    df['total_hours_open'] = hours
    return df

def add_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean columns for each category in over10000 to the dataframe."""
    cat = df.categories
    over10000 = cat.value_counts()[cat.value_counts() > 1000]
    for category in over10000.index: 
        df[category] = df.categories.str.contains(category)
    return df


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
