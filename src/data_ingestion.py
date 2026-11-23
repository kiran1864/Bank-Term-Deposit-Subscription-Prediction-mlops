from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/bank.csv")
OUTPUT_PATH = Path("data/ingested/bank.csv")
TARGET = "deposit"


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_PATH}. Put the Bank CSV at this location."
        )

    data = pd.read_csv(RAW_PATH)

    if data.empty:
        raise ValueError("The input dataset is empty.")

    if TARGET not in data.columns:
        raise ValueError(f"Target column '{TARGET}' is missing.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)

    print(f"Data ingestion completed: {data.shape[0]} rows, {data.shape[1]} columns")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()