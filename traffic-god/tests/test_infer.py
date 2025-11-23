from pathlib import Path

from src.perception.infer import InferenceConfig, flatten_trajectories


def test_flatten_trajectories_round_trip():
    history = {1: [(1, 10.0, 5.0), (2, 11.0, 5.5)], 2: [(1, 0.0, 0.0)]}
    rows = flatten_trajectories(history)
    assert len(rows) == 3
    assert rows[0] == {"track_id": 1, "frame": 1, "x": 10.0, "y": 5.0}


def test_inference_config_defaults(tmp_path):
    cfg = InferenceConfig(video_path=tmp_path / "vid.mp4", output_csv=tmp_path / "out.csv")
    assert cfg.model_path == "yolov8n.pt"
    assert cfg.output_csv.parent == tmp_path
