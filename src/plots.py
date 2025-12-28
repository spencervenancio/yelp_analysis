from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
import matplotlib.pyplot as plt

from src.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

def plot_coefficients(model, feature_names, start: int = 0, end: int = 20):
    coeff = model.coef_[0]
    coef_df = pd.DataFrame({'feature': feature_names, 'coefficient': coeff})
    # sort by absolute magnitude so the largest contributors appear first
    coef_df['abs_coefficient'] = coef_df['coefficient'].abs()
    coef_df = coef_df.sort_values(by='abs_coefficient', ascending=False)
    selected = coef_df.iloc[start:end]

    print(selected[['feature', 'coefficient']].to_dict('records'))

    plt.figure(figsize=(10, 6))
    # reverse so the largest bar appears at the top of the horizontal bar chart
    plt.barh(selected['feature'][::-1], selected['coefficient'][::-1])
    plt.xlabel('Coefficient Value')
    plt.title('Regression Coefficients')
    plt.tight_layout()
    plt.show()

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating plot from data...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Plot generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
