import os
import torch
import pytest
from torch.utils.data import TensorDataset, DataLoader
from src.model import ScratchCNN
from src.utils import CLASS_NAMES, load_saved_evaluation_assets

@pytest.fixture
def dummy_model():
    return ScratchCNN(num_classes=4)

def test_model_forward_shape(dummy_model):
    """Test that a batch input (Batch, Channels, Height, Width) outputs (Batch, Num_Classes)."""
    x = torch.randn(2, 3, 128, 128)
    out = dummy_model(x)
    assert out.shape == (2, 4), f"Expected shape (2, 4), got {out.shape}"

def test_model_predict_dummy(dummy_model, tmp_path):
    """Test inference method outputs with a dummy PIL image."""
    from PIL import Image

    img = Image.new("RGB", (128, 128), color="red")
    # mock mean and std for testing
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    pred, confidence, prob_dict = dummy_model.predict(img, mean, std)
    
    assert pred in CLASS_NAMES
    assert 0.0 <= confidence <= 100.0
    assert len(prob_dict) == 4
    
def test_model_evaluation(dummy_model, tmp_path):    
    """Test saving and reading evaluation assets to a target directory."""
    device = torch.device("cpu")

    # Mock DataLoader with synthetic inputs
    dummy_x = torch.randn(10, 3, 128, 128)
    dummy_y = torch.randint(0, 4, (10,))
    loader = DataLoader(TensorDataset(dummy_x, dummy_y), batch_size=5)

    # Execute evaluation pipeline to output directory
    output_dir = str(tmp_path / "summary")
    dummy_model.run_and_save_evaluation(
            test_loader=loader,
            output_dir=output_dir
            )

    # Verify assets were created and are loadable
    fig_path, metrics_df, summary = load_saved_evaluation_assets(output_dir=output_dir)
    
    assert os.path.exists(fig_path)
    assert metrics_df is not None
    assert summary["accuracy"] >= 0.0
