from pathlib import Path

import pytest

from model import Model
from load_model import ModelLoadError, discover_models, load_model_class


def write_model(directory, module_name, source):
    """Write a temporary trusted model module."""
    model_path = directory / f"{module_name}.py"
    model_path.write_text(source, encoding="utf-8")
    return model_path


@pytest.mark.smoke
def test_discover_models_returns_sorted_valid_neighbor_modules(tmp_path):
    """List valid model modules from only the selected script directory."""
    write_model(tmp_path, "model_zeta", "class Model_zeta: pass\n")
    write_model(tmp_path, "model_alpha", "class Model_alpha: pass\n")
    write_model(tmp_path, "model_bad-name", "class Invalid: pass\n")
    (tmp_path / "other.py").write_text("", encoding="utf-8")

    assert discover_models(tmp_path) == ["model_alpha", "model_zeta"]


@pytest.mark.smoke
def test_load_model_class_requires_fixed_name_and_model_subclass(tmp_path):
    """Load the expected local class and reject invalid model contracts."""
    write_model(
        tmp_path,
        "model_custom",
        "from model import Model\nclass Model_custom(Model):\n    pass\n",
    )
    write_model(tmp_path, "model_missing", "class Other: pass\n")
    write_model(tmp_path, "model_invalid", "class Model_invalid: pass\n")

    model_class = load_model_class("model_custom", tmp_path)

    assert model_class.__name__ == "Model_custom"
    assert issubclass(model_class, Model)
    with pytest.raises(ModelLoadError, match="Model_missing"):
        load_model_class("model_missing", tmp_path)
    with pytest.raises(ModelLoadError, match="subclass"):
        load_model_class("model_invalid", tmp_path)


@pytest.mark.smoke
def test_load_model_class_rejects_paths_and_reloads_source(tmp_path):
    """Reject path-like identifiers and execute fresh source on every load."""
    model_path = write_model(
        tmp_path,
        "model_reload",
        "from model import Model\nclass Model_reload(Model):\n    version = 1\n",
    )
    first_class = load_model_class("model_reload", tmp_path)
    model_path.write_text(
        "from model import Model\nclass Model_reload(Model):\n    version = 2\n",
        encoding="utf-8",
    )
    second_class = load_model_class("model_reload", tmp_path)

    assert first_class.version == 1
    assert second_class.version == 2
    assert first_class is not second_class
    with pytest.raises(ModelLoadError, match="identifier"):
        load_model_class("../model_reload", tmp_path)


@pytest.mark.smoke
def test_project_discovery_includes_default_model():
    """Discover the bundled default model beside the executable scripts."""
    project_root = Path(__file__).resolve().parents[1]

    assert "model_base" in discover_models(project_root)
    assert "model_base_fast" in discover_models(project_root)
    assert load_model_class("model_base_fast", project_root).__name__ == "Model_base_fast"
    assert "model_base_fast_fixed_fecundity" in discover_models(project_root)
    assert (
        load_model_class("model_base_fast_fixed_fecundity", project_root).__name__
        == "Model_base_fast_fixed_fecundity"
    )
    assert "model_base_z" in discover_models(project_root)
    assert load_model_class("model_base_z", project_root).__name__ == "Model_base_z"
    assert "model_base_diploid" in discover_models(project_root)
    assert (
        load_model_class("model_base_diploid", project_root).__name__
        == "Model_base_diploid"
    )
    assert "model_alleles" in discover_models(project_root)
    assert load_model_class("model_alleles", project_root).__name__ == "Model_alleles"


@pytest.mark.smoke
def test_bundled_models_expose_explicit_structured_descriptions():
    """Require a custom purpose, inheritance, and difference description per model."""
    project_root = Path(__file__).resolve().parents[1]
    expected_titles = {
        "model_base": "# Baseline beta/Gompertz model",
        "model_base_fast": "# Fast beta/Gompertz model",
        "model_base_fast_fixed_fecundity": "# Fast beta model with fixed parent fecundity",
        "model_base_z": "# Fast fixed-fecundity beta model with mutation sign bias Z",
        "model_base_diploid": "# Diploid beta model",
        "model_alleles": "# Multi-locus diploid beta model",
    }

    placeholder = Model.description()
    assert "Replace this text" in placeholder
    assert "## Inheritance" in placeholder
    for model_name, expected_title in expected_titles.items():
        model_class = load_model_class(model_name, project_root)
        description = model_class.description()
        assert "description" in model_class.__dict__
        assert description.startswith(expected_title)
        assert "## Purpose" in description
        assert "## Inheritance" in description
        assert "## Difference from its parent" in description


@pytest.mark.smoke
def test_load_model_class_rejects_invalid_metadata(tmp_path):
    """Wrap strict metadata failures in the loader's public error type."""
    write_model(
        tmp_path,
        "model_bad_metadata",
        "from model import Model\n"
        "class Model_bad_metadata(Model):\n"
        "    @staticmethod\n"
        "    def add_graphs():\n"
        "        return [{'filename': 'bad', 'values': []}]\n",
    )

    with pytest.raises(ModelLoadError, match="model_bad_metadata.*values"):
        load_model_class("model_bad_metadata", tmp_path)