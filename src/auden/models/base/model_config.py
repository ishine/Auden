"""
Configuration utilities for auden models.

This module provides a lightweight, inheritance-friendly ``BaseConfig`` class
that model-specific configs can extend. It is intentionally similar in spirit
to Hugging Face's ``PretrainedConfig`` to provide a predictable developer
experience while keeping implementation minimal.

Key features:
- Convert to a Python dict (including nested configs implementing ``to_dict()``)
- Save to and load from JSON files (``config.json``)
- Optional backup of existing config files when overwriting
- Convenience helpers: ``to_json_string()``, ``to_json_file()``, ``from_dict()``,
  ``update()``, and simple dict-like accessors (``keys()``, ``get()``,
  ``__contains__``)

Notes:
- The field ``model_type`` is REQUIRED for subclasses. It must be a non-empty
  string so that ``AutoConfig`` / ``AutoModel`` can resolve the correct classes.
- By default, arbitrary keyword fields are accepted and stored; concrete
  config classes can override this behavior if strictness is desired.
"""

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path


class BaseConfig:
    model_type: str = None

    def __init__(self, **kwargs):
        """Initialize a config instance.

        Behavior mirrors common ML config bases:
        - Class attributes become default values.
        - Any provided keyword arguments override defaults and are stored as
          attributes, even if they were not declared ahead of time. This keeps
          configs forward-compatible with newer versions.
        """
        for key, value in self.__class__.__dict__.items():
            if not key.startswith("_") and not callable(value):
                setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.__class__ is not BaseConfig:
            model_type_value = getattr(self, "model_type", None)
            if not isinstance(model_type_value, str) or model_type_value.strip() == "":
                raise ValueError(
                    "'model_type' must be set to a non-empty string in subclasses of BaseConfig."
                )

    def to_dict(self):
        """Return a JSON-serializable dict representation of this config.

        Nested config objects are supported if they implement a ``to_dict()``
        method. Private attributes (starting with ``_``) and callables are
        skipped.
        """
        output = {}
        for key, value in self.__dict__.items():
            if key.startswith("_") or callable(value):
                continue
            if hasattr(value, "to_dict"):
                output[key] = value.to_dict()
            else:
                output[key] = value
        return output

    def save_pretrained(self, output_dir: str):
        """Save this config instance to ``output_dir/config.json``.

        Behavior:
        - If an identical config already exists, saving is skipped.
        - If a different config exists, it is backed up with a timestamp before saving.
        """
        new_config = self.to_dict()
        os.makedirs(output_dir, exist_ok=True)
        config_path = Path(output_dir) / "config.json"

        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    existing_config = json.load(f)
                if existing_config == new_config:
                    logging.info(
                        f"[save_config] Skipped saving. Config identical to existing one."
                    )
                    return
                # Backup old config
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = config_path.with_name(f"config.{timestamp}.bak.json")
                shutil.move(config_path, backup_path)
                logging.info(
                    f"[save_config] Existing config backed up to: {backup_path}"
                )
            except Exception as e:
                logging.warning(
                    f"[save_config] Could not compare with existing config: {e}. Proceeding to save."
                )

        # Save new config
        with open(config_path, "w") as f:
            json.dump(new_config, f, indent=2)
        logging.info(f"[save_config] Saved config to: {config_path}")

    @classmethod
    def from_pretrained(cls, config_path: str):
        """Load a config from a directory or a JSON file.

        Args:
            config_path: A directory containing ``config.json`` or a path to a JSON file.

        Returns:
            An instance of ``cls`` initialized with values loaded from JSON.
        """
        if os.path.isdir(config_path):
            config_file = os.path.join(config_path, "config.json")
        else:
            config_file = config_path
        with open(config_file, "r") as f:
            data = json.load(f)
        return cls(**data)

    def to_json_string(self, indent: int = 2) -> str:
        """Return a JSON string representation of this config.

        Args:
            indent: Indentation level for pretty printing.
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_json_file(self, json_file: str, indent: int = 2) -> None:
        """Write this config to a JSON file.

        Args:
            json_file: File path to write to.
            indent: Indentation level for pretty printing.
        """
        with open(json_file, "w") as f:
            f.write(self.to_json_string(indent=indent))

    @classmethod
    def from_dict(cls, data: dict):
        """Instantiate config from a Python dict."""
        return cls(**data)

    def update(self, values: dict):
        """Update config values in-place from a dict and return self."""
        for key, value in values.items():
            setattr(self, key, value)
        return self

    # Dict-like conveniences
    def keys(self):
        """Return public attribute names treated as config keys."""
        return [k for k in self.__dict__.keys() if not k.startswith("_")]

    def get(self, key, default=None):
        """Get a config value with a default, similar to dict.get."""
        return getattr(self, key, default)

    def __contains__(self, item: object) -> bool:
        return (
            hasattr(self, item) and not item.startswith("_")
            if isinstance(item, str)
            else False
        )

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        return f"{cls_name}({self.to_json_string(indent=2)})"
