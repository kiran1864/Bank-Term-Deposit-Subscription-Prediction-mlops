from pathlib import Path
import pandas as pd

TRAIN_INPUT = Path("data/processed/train.csv")
TEST_INPUT = Path("data/processed/test.csv")

X_TRAIN_PATH = Path("data/processed/X_train.csv")
X_TEST_PATH = Path("data/processed/X_test.csv")
Y_TRAIN_PATH = Path("data/processed/y_train.csv")
Y_TEST_PATH = Path("data/processed/y_test.csv")

TARGET = "deposit"


def separate_features_and_target(input_path):
    data = pd.read_csv(input_path)

    if TARGET not in data.columns:
        raise ValueError(f"Target column '{TARGET}' is missing in {input_path}")

    # Numeric scaling and categorical one-hot encoding remain inside the
    # sklearn ColumnTransformer/Pipeline during training to prevent leakage.
    X = data.drop(columns=[TARGET]).copy()
    y = data[TARGET].astype(int).copy()
    return X, y


def main():
    X_train, y_train = separate_features_and_target(TRAIN_INPUT)
    X_test, y_test = separate_features_and_target(TEST_INPUT)

    X_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(X_TRAIN_PATH, index=False)
    X_test.to_csv(X_TEST_PATH, index=False)
    y_train.to_frame(name=TARGET).to_csv(Y_TRAIN_PATH, index=False)
    y_test.to_frame(name=TARGET).to_csv(Y_TEST_PATH, index=False)

    print("Feature and target separation completed.")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"Saved split files under: {X_TRAIN_PATH.parent}")


if __name__ == "__main__":
    main()