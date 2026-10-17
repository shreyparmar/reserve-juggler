
import pandas as pd
import pytest

from reserve_juggler.data_loader import split_dataset


def make_sample_data(rows: int = 20) -> pd.DataFrame:
    """Build a small hourly dataset for testing the split."""

    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=rows,
                freq="h",
            ),
            "row_id": range(rows),
        }
    )


def test_split_has_expected_sizes():
    data = make_sample_data(20)

    training, validation, test = split_dataset(data)

    assert len(training) == 14
    assert len(validation) == 3
    assert len(test) == 3


def test_split_preserves_chronological_order():
    data = make_sample_data(20)

    training, validation, test = split_dataset(data)

    assert training["row_id"].tolist() == list(range(14))
    assert validation["row_id"].tolist() == list(range(14, 17))
    assert test["row_id"].tolist() == list(range(17, 20))

    assert training["timestamp"].max() < validation["timestamp"].min()
    assert validation["timestamp"].max() < test["timestamp"].min()


def test_split_rejects_invalid_timestamps():
    data = make_sample_data(20)

    # Introduce an out-of-order timestamp.
    data.loc[5, "timestamp"] = (
        data.loc[4, "timestamp"] - pd.Timedelta(hours=1)
    )

    with pytest.raises(ValueError):
        split_dataset(data)
