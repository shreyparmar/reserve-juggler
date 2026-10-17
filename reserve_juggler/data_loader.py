
from pathlib import Path

import pandas as pd


# These ratios produce 6,132 / 1,314 / 1,314 rows
# when the input contains 8,760 hourly observations.
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15

REQUIRED_COLUMNS = [
    "timestamp",
    "deposits_usd",
    "withdrawals_usd",
    "net_flow_usd",
    "repo_sweep_window_open",
    "repo_sweep_mature",
    "tier1_repo_rate_apy",
    "tier2_bond_yield_pct",
    "tier2_bond_price_index",
    "tier2_bond_mod_duration",
    "tier3_stock_index_price",
    "tier3_stock_volatility_pct",
    "tier1_early_redemption_haircut_pct",
    "tier2_bond_liquidation_haircut_pct",
    "tier3_stock_liquidation_haircut_pct",
    "emergency_overdraft_penalty_apr_pct",
]

NUMERIC_COLUMNS = [
    column
    for column in REQUIRED_COLUMNS
    if column not in {
        "timestamp",
        "repo_sweep_window_open",
        "repo_sweep_mature",
    }
]

BOOLEAN_COLUMNS = [
    "repo_sweep_window_open",
    "repo_sweep_mature",
]


def _validate_timestamps(frame: pd.DataFrame) -> pd.Series:
    """Parse timestamps and require unique, continuous hourly records."""

    timestamps = pd.to_datetime(
        frame["timestamp"],
        errors="coerce",
    )

    if timestamps.isna().any():
        raise ValueError("Dataset contains invalid or missing timestamps.")

    if timestamps.duplicated().any():
        raise ValueError("Dataset contains duplicate timestamps.")

    if not timestamps.is_monotonic_increasing:
        raise ValueError(
            "Timestamps must be in strictly ascending chronological order."
        )

    if len(timestamps) < 2:
        raise ValueError("At least two hourly observations are required.")

    intervals = timestamps.diff().iloc[1:]

    if not intervals.eq(pd.Timedelta(hours=1)).all():
        raise ValueError(
            "Timestamps must be exactly one hour apart with no gaps."
        )

    return timestamps


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate the source treasury CSV."""

    csv_path = Path(path)

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path.resolve()}"
        )

    frame = pd.read_csv(csv_path)

    missing_columns = sorted(
        set(REQUIRED_COLUMNS) - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    frame["timestamp"] = _validate_timestamps(frame)

    # Convert transaction, rate, price and haircut fields to numbers.
    for column in NUMERIC_COLUMNS:
        converted = pd.to_numeric(frame[column], errors="coerce")

        if converted.isna().any():
            raise ValueError(
                f"Column '{column}' contains missing or non-numeric values."
            )

        frame[column] = converted

    # Accept standard CSV boolean values, including true/false and 1/0.
    boolean_values = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }

    for column in BOOLEAN_COLUMNS:
        converted = (
            frame[column]
            .astype("string")
            .str.strip()
            .str.lower()
            .map(boolean_values)
        )

        if converted.isna().any():
            raise ValueError(
                f"Column '{column}' must contain valid boolean values."
            )

        frame[column] = converted.astype(bool)

    if frame.isna().any().any():
        raise ValueError("Dataset contains missing cells.")

    if (frame["deposits_usd"] < 0).any():
        raise ValueError("Deposit amounts cannot be negative.")

    if (frame["withdrawals_usd"] < 0).any():
        raise ValueError("Withdrawal amounts cannot be negative.")

    # Permit a two-cent difference for rounding.
    calculated_flow = (
        frame["deposits_usd"] - frame["withdrawals_usd"]
    )

    mismatch = (
        calculated_flow - frame["net_flow_usd"]
    ).abs() > 0.02

    if mismatch.any():
        raise ValueError(
            "net_flow_usd does not match deposits_usd - withdrawals_usd."
        )

    return frame


def split_dataset(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split chronologically into training, validation and test sets.

    No shuffling is performed. Test observations remain the final
    observations in the original time series.
    """

    if "timestamp" not in frame.columns:
        raise ValueError("Dataset must contain a timestamp column.")

    data = frame.copy()
    data["timestamp"] = _validate_timestamps(data)

    row_count = len(data)

    train_end = int(row_count * TRAIN_FRACTION)
    validation_end = (
        train_end + int(row_count * VALIDATION_FRACTION)
    )

    if train_end < 1:
        raise ValueError("Not enough data for a training set.")

    if validation_end >= row_count:
        raise ValueError(
            "Not enough data to create three non-empty sets."
        )

    training = data.iloc[:train_end].reset_index(drop=True)
    validation = (
        data.iloc[train_end:validation_end]
        .reset_index(drop=True)
    )
    test = data.iloc[validation_end:].reset_index(drop=True)

    return training, validation, test
