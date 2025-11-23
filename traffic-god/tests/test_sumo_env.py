from src.sim.sumo_env import SumoIntersectionEnv, build_env_config


def test_build_env_config_override():
    cfg = build_env_config({"max_steps": 123})
    assert cfg["max_steps"] == 123
    assert cfg["reward_weights"]["queue"] == -0.1


def test_env_step_shapes():
    env = SumoIntersectionEnv({"max_steps": 2}, dry_run=True)
    obs, _ = env.reset()
    assert obs.shape == (8,)
    obs, reward, done, _, info = env.step(0)
    assert obs.shape == (8,)
    assert isinstance(reward, float)
    assert "congestion_proxy" in info
    env.close()
