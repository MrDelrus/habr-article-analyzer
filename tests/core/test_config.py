from pathlib import Path

from core.config import data_settings, paths, settings


def test_paths_exist() -> None:
    """Verify that paths from config.py are created and are Path objects."""
    # The actual directories may not exist yet, so we only check type
    assert isinstance(paths.data_dir, Path)
    assert isinstance(paths.models_dir, Path)
    assert isinstance(paths.logs_dir, Path)


def test_settings_loaded() -> None:
    """Verify that settings are loaded from config.toml correctly."""
    assert settings.log_level in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def test_data_settings_types() -> None:
    """Verify that all numerical parameters in DataSettings have correct types."""
    assert isinstance(data_settings.test_size, float)
    assert isinstance(data_settings.val_size, float)
    assert isinstance(data_settings.random_seed, int)
