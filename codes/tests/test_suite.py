import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path
import json
import cv2
import sys

# Add project root directory to path if running directly
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from codes.unified_data import Config, MultiDirectoryDataLoader
from codes.unified_training import ThermalFaceDetector
from codes.model_registry import create_model

# Import models to ensure they are registered
import codes.unet_v2
import codes.transunet
import codes.swin_unet_plus_plus

@pytest.mark.parametrize("model_name,batch_size", [
    ("unet", 2),
    ("transunet", 2),
    ("swin_unet_plus_plus", 1), # SwinUNet++ is heavy, test with batch size 1
])
def test_model_shapes(model_name, batch_size):
    kwargs = {
        'in_channels': 1,
        'num_classes': 10
    }
    if model_name in ['transunet', 'swin_unet_plus_plus']:
        kwargs['img_size'] = 256
        
    model = create_model(model_name, **kwargs)
    x = torch.randn(batch_size, 1, 256, 256)
    y = model(x)
    assert y.shape == (batch_size, 10, 256, 256)

@pytest.fixture
def mock_config():
    return Config()

@pytest.fixture
def mock_model_path(tmp_path):
    model = create_model("unet", in_channels=1, num_classes=10)
    model_path = tmp_path / "dummy_model.pth"
    torch.save(model.state_dict(), model_path)
    return model, model_path

def test_detector_initialization_and_prediction(mock_config, mock_model_path):
    model, model_path = mock_model_path
    detector = ThermalFaceDetector(model, str(model_path), mock_config)
    
    mock_img = np.random.uniform(30.0, 40.0, (240, 320)).astype(np.float32)
    
    regions, pred_mask = detector.predict(mock_img)
    
    assert pred_mask.shape == mock_img.shape
    for region_name in mock_config.REGION_NAMES:
        assert region_name in regions
        assert regions[region_name].shape == mock_img.shape
        assert np.all((regions[region_name] == 0) | (regions[region_name] == 1))
        
    stats = detector.get_stats_info(mock_img, regions)
    for region_name in mock_config.REGION_NAMES:
        assert region_name in stats
        r_stats = stats[region_name]
        assert 'mean' in r_stats
        assert 'median' in r_stats
        assert 'mode' in r_stats
        assert 'std' in r_stats
        assert 'pixel_count' in r_stats
        
        if r_stats['pixel_count'] > 0:
            assert r_stats['mean'] is not None
            assert 30.0 <= r_stats['mean'] <= 40.0

@pytest.fixture
def mock_data_dir(tmp_path):
    original_data_dir = Config.DATA_DIR
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    Config.DATA_DIR = data_dir
    
    s1_dir = data_dir / "S1"
    s1_dir.mkdir()
    
    tiff_path = s1_dir / "R11104.tiff"
    dummy_img = np.ones((100, 100), dtype=np.uint16) * 31000
    cv2.imwrite(str(tiff_path), dummy_img)
    
    polygons = {
        "R11104": {
            "Nariz": [[10, 10], [10, 20], [20, 20], [20, 10]],
            "Olho esquerdo": [[30, 30], [30, 40], [40, 40], [40, 30]]
        }
    }
    with open(data_dir / "S1_polygonal_masks.json", "w") as f:
        json.dump(polygons, f)
        
    with open(data_dir / "S1_bounding_boxes.csv", "w") as f:
        f.write("ID,min_x,min_y,max_x,max_y\nR11104,5,5,95,95\n")
        
    yield data_dir, tiff_path, polygons
    Config.DATA_DIR = original_data_dir

def test_discovery_and_loading(mock_data_dir, tmp_path):
    data_dir, tiff_path, polygons = mock_data_dir
    output_dir = tmp_path / "outputs"
    log_dir = tmp_path / "logs"
    config = Config(output_dir=str(output_dir), log_dir=str(log_dir))
    
    loader = MultiDirectoryDataLoader(config)
    loader.load_annotations()
    
    assert "S1" in loader.datasets
    assert "S1/R11104" in loader.all_polygons
    assert loader.all_polygons["S1/R11104"]["Nariz"] == polygons["R11104"]["Nariz"]
    
    assert loader.all_bboxes is not None
    assert "S1/R11104" in loader.all_bboxes["ID"].values
    
    found_tiff_path = loader.get_tiff_path("S1/R11104")
    assert Path(found_tiff_path) == tiff_path
    
    img = loader.load_thermal_image("S1/R11104")
    assert img.shape == (100, 100)
    assert np.isclose(img[0, 0], 36.85, atol=0.01)
