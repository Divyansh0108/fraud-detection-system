import pandas as pd
import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler
from src.config import settings


def load_data():
    """Loads raw data from the DVC-tracked path."""
    print(f"Loading data from {settings.RAW_DATA_PATH}...")
    return pd.read_csv(settings.RAW_DATA_PATH)


def split_and_process(df: pd.DataFrame):
    """
    1. Splits data FIRST to prevent leakage.
    2. Fits scaler on Train, transforms Train & Test.
    3. Saves the scaler artifact for production.
    """
    print("Splitting data (Stratified)...")

    # 1. Stratified Split
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=settings.TEST_SIZE, random_state=settings.RANDOM_STATE
    )

    for train_index, test_index in sss.split(df, df[settings.TARGET_COLUMN]):
        train_df = df.iloc[train_index].copy()
        test_df = df.iloc[test_index].copy()

    # 2. Define Features to Scale
    scale_cols = ["Time", "Amount"]

    # 3. Fit Scaler ONLY on Train
    print("Fitting RobustScaler on Train set only...")
    scaler = RobustScaler()

    # We fit on the specific columns of the training data
    scaler.fit(train_df[scale_cols])

    # 4. Transform both Train and Test
    train_df[scale_cols] = scaler.transform(train_df[scale_cols])
    test_df[scale_cols] = scaler.transform(test_df[scale_cols])

    # 5. Save the Scaler (We need this for the API later!)
    settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    scaler_path = settings.MODEL_DIR / "preprocessor.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler saved to {scaler_path}")

    # 6. Save Processed Data
    settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(settings.PROCESSED_DATA_DIR / "train.csv", index=False)
    test_df.to_csv(settings.PROCESSED_DATA_DIR / "test.csv", index=False)

    return train_df, test_df


if __name__ == "__main__":
    df = load_data()
    train_df, test_df = split_and_process(df)

    print(f"✅ Data processing complete (No Leakage).")
    print(f"   Train shape: {train_df.shape}")
    print(f"   Test shape:  {test_df.shape}")
