"""Discover and load trusted dynamic model modules."""

from pathlib import Path
import re
import sys
from types import ModuleType

from model import Model
from model_metadata import ModelMetadataError, validate_model_metadata


MODEL_NAME_PATTERN = re.compile(r"model_[A-Za-z0-9_]+\Z")
DEFAULT_MODEL_DIRECTORY = Path(__file__).resolve().parent


class ModelLoadError(RuntimeError):
    """Report an invalid model identifier, module, or class contract."""


def _resolve_model_directory(model_directory):
    """Return the selected trusted model directory as an absolute path."""
    directory = DEFAULT_MODEL_DIRECTORY if model_directory is None else model_directory
    return Path(directory).resolve()


def discover_models(model_directory=None):
    """Return sorted valid model module names from one trusted directory."""
    directory = _resolve_model_directory(model_directory)
    if not directory.is_dir():
        return []

    return sorted(
        path.stem
        for path in directory.glob("model_*.py")
        if path.is_file() and MODEL_NAME_PATTERN.fullmatch(path.stem)
    )


def _execute_fresh_module(module_name, module_path, model_directory):
    """Compile and execute trusted source without using the import cache."""
    module = ModuleType(module_name)
    module.__file__ = str(module_path)
    source = module_path.read_text(encoding="utf-8")
    code = compile(source, str(module_path), "exec")

    directory_text = str(model_directory)
    added_to_path = directory_text not in sys.path
    if added_to_path:
        sys.path.insert(0, directory_text)
    try:
        exec(code, module.__dict__)
    finally:
        if added_to_path:
            sys.path.remove(directory_text)
    return module


def load_model_class(module_name, model_directory=None):
    """Fresh-load and return the fixed-name Model subclass from a trusted file."""
    if not isinstance(module_name, str) or not MODEL_NAME_PATTERN.fullmatch(module_name):
        raise ModelLoadError(f"Invalid model identifier: {module_name!r}")

    directory = _resolve_model_directory(model_directory)
    module_path = directory / f"{module_name}.py"
    if not module_path.is_file():
        raise ModelLoadError(f"Model file not found: {module_path}")

    try:
        module = _execute_fresh_module(module_name, module_path, directory)
    except Exception as error:
        raise ModelLoadError(f"Failed to load model {module_name}: {error}") from error

    class_name = module_name[0].upper() + module_name[1:]
    model_class = getattr(module, class_name, None)
    if not isinstance(model_class, type):
        raise ModelLoadError(f"Model module must define class {class_name}")
    if not issubclass(model_class, Model):
        raise ModelLoadError(f"{class_name} must be a subclass of Model")
    try:
        validate_model_metadata(model_class)
    except ModelMetadataError as error:
        raise ModelLoadError(f"Invalid metadata for {module_name}: {error}") from error
    return model_class