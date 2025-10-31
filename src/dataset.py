from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import json
import pandas as pd

from src.config import RAW_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()

def read_json(data_path):
    with open(data_path) as f:
        data = [json.loads(line) for line in f]
    return pd.DataFrame(data)
    
tables = ['review', 'checkin', 'business', 'user', 'tip']

@app.command()
def main(
    output_dir: Path = INTERIM_DATA_DIR,
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    for table in tqdm(tables, total=5):
            logger.info(f"Processing table: {table}")
            df = read_json(RAW_DATA_DIR / f"yelp_academic_dataset_{table}.json")
            file_name = output_dir / f"{table}.parquet"
            df.to_parquet(file_name, index=False)
            
    logger.success("Processing dataset complete.")
    # -----------------------------------------
if __name__ == "__main__":
    app()
