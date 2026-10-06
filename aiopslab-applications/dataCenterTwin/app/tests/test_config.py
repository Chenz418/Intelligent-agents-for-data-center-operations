from pathlib import Path

import pytest

from dc_twin.config import default_config, load_config


def test_load_config_without_path_uses_defaults(monkeypatch):
    monkeypatch.delenv("DC_TWIN_CONFIG", raising=False)

    config = load_config()

    assert config == default_config()
    assert config.simulation.auto_advance is False


def test_load_config_missing_explicit_path_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("DC_TWIN_CONFIG", raising=False)
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="Data Center Twin configuration file not found"):
        load_config(missing_path)


def test_load_config_missing_env_path_raises(tmp_path, monkeypatch):
    missing_path = tmp_path / "missing-env.yaml"
    monkeypatch.setenv("DC_TWIN_CONFIG", str(missing_path))

    with pytest.raises(FileNotFoundError, match=str(Path(missing_path))):
        load_config()
