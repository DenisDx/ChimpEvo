import pytest

from model import Model
from model_base import Model_base
from model_metadata import ModelMetadataError, validate_model_metadata


@pytest.mark.smoke
def test_generic_model_metadata_uses_normalized_defaults():
    """Normalize generic settings, values, and age graph declarations."""
    metadata = validate_model_metadata(Model)

    assert metadata["settings"]["seed"] == {
        "description": "Random seed; zero selects a random seed",
        "default": 0,
        "type": "int",
    }
    assert metadata["values"]["count"] == {
        "title": "Population",
        "description": "",
        "annual": True,
        "final": True,
        "format": "d",
    }
    assert metadata["graphs"][0]["filename"] == "age_distribution"
    assert metadata["graphs"][0]["values"] == ["age"]
    assert metadata["graphs"][0]["type"] == "distr"
    assert metadata["graphs"][0]["annual"] is True
    assert metadata["graphs"][0]["final"] is True
    assert metadata["graphs"][0]["animated"] is True
    assert metadata["metagraphs"] == []


@pytest.mark.smoke
def test_model_base_declares_current_biological_settings():
    """Expose existing default-model settings with types and inclusive ranges."""
    settings = validate_model_metadata(Model_base)["settings"]

    assert settings["initial_population"] == {
        "description": "Starting population count",
        "default": 2000,
        "type": "int",
        "min": 1,
        "max": 100000000,
    }
    assert settings["beta_initial"] == {
        "description": "Initial genetic parameter",
        "default": 0.11,
        "type": "float",
        "min": 0.0,
        "max": 1.0,
    }
    assert settings["seed"] == Model.add_settings()["seed"]


@pytest.mark.smoke
def test_model_base_declares_beta_distribution_and_time_graphs():
    """Expose age and beta dynamic graph declarations for the default model."""
    graphs = validate_model_metadata(Model_base)["graphs"]

    assert [graph["filename"] for graph in graphs] == [
        "age_distribution", "beta_distribution", "beta_evolution",
    ]
    assert graphs[1]["type"] == "distr"
    assert graphs[1]["values"] == ["beta"]
    assert graphs[2]["type"] == "time"
    assert graphs[2]["values"] == ["avg_beta", "avg_beta_ema"]


@pytest.mark.smoke
def test_model_base_declares_beta_mutation_metagraph():
    """Expose final beta lines over the model's default batch sweep setting."""
    metagraph = validate_model_metadata(Model_base)["metagraphs"][0]

    assert metagraph["filename"] == "beta_by_mutation"
    assert metagraph["xvalue"] == "mutation_x"
    assert metagraph["values"] == ["avg_beta", "avg_beta_ema"]
    assert metagraph["animated"] is True


@pytest.mark.smoke
def test_graph_metadata_normalizes_defaults_and_references():
    """Normalize graph defaults and validate scalar and field references."""

    class GraphModel(Model):
        """Declare one time graph and one distribution graph."""

        @staticmethod
        def add_graphs():
            """Return valid graph declarations with omitted defaults."""
            return [
                {"filename": "population", "type": "time", "values": ["count"]},
                {"filename": "ages", "values": ["age"], "bin_count": 25},
            ]

    graphs = validate_model_metadata(GraphModel)["graphs"]

    assert graphs[0]["type"] == "time"
    assert graphs[0]["labels"] == ["count"]
    assert graphs[0]["animated"] is True
    assert graphs[1]["type"] == "distr"
    assert graphs[1]["bin_count"] == 25
    assert graphs[1]["scale"] == 1.0


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("method_name", "declaration", "message"),
    [
        ("add_settings", {"x": {"default": 1}}, "description"),
        ("add_settings", {"x": {"description": "x", "default": 1, "type": "number"}}, "type"),
        ("add_values", {"x": {"unknown": True}}, "unknown"),
        ("add_graphs", [{"filename": "x", "values": []}], "values"),
        ("add_graphs", [{"filename": "x", "values": ["age"], "unknown": True}], "unknown"),
        ("add_graphs", [{"filename": "x", "values": ["age"], "annual": False, "final": False}], "annual"),
        ("add_graphs", [{"filename": "x", "values": ["age"], "labels": []}], "labels"),
        ("add_metagraphs", [{"filename": "x", "values": ["missing"]}], "missing"),
        ("add_metagraphs", [{"filename": "x", "values": ["count"], "labels": []}], "labels"),
        ("add_metagraphs", [{"filename": "x", "values": ["count"], "unknown": True}], "unknown"),
    ],
)
def test_metadata_rejects_invalid_declarations(method_name, declaration, message):
    """Reject malformed or meaningless strict metadata declarations."""

    class InvalidModel(Model):
        """Return one injected invalid declaration."""

    setattr(InvalidModel, method_name, staticmethod(lambda: declaration))

    with pytest.raises(ModelMetadataError, match=message):
        validate_model_metadata(InvalidModel)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "graphs",
    [
        [{"filename": "x", "type": "time", "values": ["missing"]}],
        [{"filename": "x", "type": "distr", "values": ["missing"]}],
    ],
)
def test_graph_metadata_rejects_unknown_value_references(graphs):
    """Reject graph series absent from the corresponding namespace."""

    class InvalidGraphModel(Model):
        """Return a graph with an unknown series."""

        @staticmethod
        def add_graphs():
            """Return the invalid graph list."""
            return graphs

    with pytest.raises(ModelMetadataError, match="missing"):
        validate_model_metadata(InvalidGraphModel)