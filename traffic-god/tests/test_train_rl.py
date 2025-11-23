from src.control.train_rl import build_training_config, train


def test_build_training_config_override():
    cfg = build_training_config({"learning_rate": 1e-4})
    assert cfg["learning_rate"] == 1e-4
    assert cfg["policy"] == "MlpPolicy"


def test_train_dry_run_metadata():
    result = train(build_training_config({"total_timesteps": 100}), dry_run=True)
    assert result["dry_run"] is True
    assert result["timesteps"] == 0
