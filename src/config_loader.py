"""
Config loader for SAM 3D GUI
Hydra-based configuration management
"""
from pathlib import Path
from omegaconf import OmegaConf
from typing import Optional


class ModelConfig:
    """Model configuration loader"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Load configuration from YAML file

        Args:
            config_path: Path to config file (default: ../config/model_config.yaml)
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "model_config.yaml"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        self.cfg = OmegaConf.load(config_path)

        # Resolve environment variables
        OmegaConf.resolve(self.cfg)

        # Store project root for resolving relative paths
        self.project_root = Path(__file__).parent.parent

    @property
    def sam2_checkpoint(self) -> str:
        """Get SAM2 checkpoint path"""
        path = Path(self.cfg.sam2.checkpoint).expanduser()
        # If relative path, resolve from project root
        if not path.is_absolute():
            path = (self.project_root / path).resolve()
        return str(path)

    @property
    def sam2_config(self) -> str:
        """Get SAM2 model config"""
        return self.cfg.sam2.config

    @property
    def sam2_device(self) -> str:
        """Get SAM2 device"""
        return self.cfg.sam2.device

    @property
    def sam3d_checkpoint_dir(self) -> str:
        """Get SAM3D checkpoint directory (try primary, then alternative)"""
        primary = Path(self.cfg.sam3d.checkpoint_dir).expanduser()
        # If relative path, resolve from project root
        if not primary.is_absolute():
            primary = (self.project_root / primary).resolve()
        if primary.exists():
            return str(primary)

        alt = Path(self.cfg.sam3d.checkpoint_dir_alt).expanduser()
        # Alt is usually absolute with ~, but check anyway
        if not alt.is_absolute():
            alt = (self.project_root / alt).resolve()
        if alt.exists():
            return str(alt)

        # Neither exists - return primary for error message
        return str(primary)

    @property
    def default_data_dir(self) -> str:
        """Get default data directory"""
        path = Path(self.cfg.data.default_dir).expanduser()
        # If relative path, resolve from project root
        if not path.is_absolute():
            path = (self.project_root / path).resolve()
        return str(path)

    @property
    def output_dir(self) -> str:
        """Get output directory"""
        path = Path(self.cfg.data.output_dir).expanduser()
        # If relative path, resolve from project root
        if not path.is_absolute():
            path = (self.project_root / path).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def get_visualization_config(self):
        """Get visualization configuration"""
        return self.cfg.visualization
