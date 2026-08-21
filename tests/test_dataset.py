import os
import pandas as pd
import pytest
from src.utils import load_dataset_stats

def test_load_dataset_stats(tmp_path):
    """Test loading mean and std from a temporary CSV file."""
    csv_file = tmp_path / "dataset_stats.csv"
    df = pd.DataFrame({
        "channel": [0, 1, 2],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    })
    df.to_csv(csv_file, index=False)

    mean, std = load_dataset_stats(str(csv_file))
    assert len(mean) == 3
    assert len(std) == 3
    assert mean[0] == pytest.approx(0.485)
