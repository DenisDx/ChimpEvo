import numpy as np
import pytest
import torch

from model import Model
from settings import DEFAULT_SETTINGS


def make_settings(**overrides):
    """Return generic model settings with optional initialization overrides."""
    return {**DEFAULT_SETTINGS, **overrides}


@pytest.mark.smoke
def test_generic_model_initializes_only_public_age_field():
    """Create an age-only population and expose age by its field name."""
    model = Model(make_settings(initial_population=6, initial_age_max=4), torch.device("cpu"))

    model.initialize_population()

    assert model.population.shape == (6, 1)
    assert model.population.dtype == torch.float32
    assert model.population_fields == {"age": 0}
    np.testing.assert_array_equal(model.get_ages(), model.get_tensor("age"))
    assert np.all((model.get_tensor("age") >= 0) & (model.get_tensor("age") <= 4))


@pytest.mark.smoke
def test_generic_model_lifecycle_has_no_births_deaths_or_stop_reason():
    """Keep the empty model alive without biological events or stop requests."""
    model = Model(make_settings(initial_population=3, initial_age_max=2), torch.device("cpu"))
    model.initialize_population()

    births = model.apply_reproduction()
    model.age_population()
    deaths = model.apply_mortality()

    assert births == 0
    assert deaths == 0
    assert model.last_born == 0
    assert model.last_death == 0
    assert model.should_stop() is None
    assert model.get_population_size() == 3


@pytest.mark.smoke
def test_get_tensor_rejects_unknown_population_field():
    """Raise a field error instead of returning an unrelated tensor column."""
    model = Model(make_settings(initial_population=1, initial_age_max=0), torch.device("cpu"))
    model.initialize_population()

    with pytest.raises(KeyError, match="unknown"):
        model.get_tensor("unknown")


@pytest.mark.smoke
def test_population_field_declarations_build_an_immutable_schema():
    """Assign columns by declaration order and prevent schema mutation."""

    class ExtendedModel(Model):
        """Declare one additional private population field."""

        @staticmethod
        def add_population_fields():
            """Return inherited fields followed by a private score field."""
            return {
                **Model.add_population_fields(),
                "score": {"public": False},
            }

    model = ExtendedModel(DEFAULT_SETTINGS.copy(), torch.device("cpu"))

    assert model.population_fields == {"age": 0, "score": 1}
    with pytest.raises(TypeError):
        model.population_fields["other"] = 2


@pytest.mark.smoke
def test_population_helpers_reject_incomplete_rows():
    """Reject initial and appended tensors that do not match schema width."""

    class TwoFieldModel(Model):
        """Declare a two-column population for row validation."""

        @staticmethod
        def add_population_fields():
            """Return age and one additional public field."""
            return {
                **Model.add_population_fields(),
                "score": {"public": True},
            }

    model = TwoFieldModel(DEFAULT_SETTINGS.copy(), torch.device("cpu"))

    with pytest.raises(ValueError, match="2 columns"):
        model._set_population(torch.zeros((3, 1)))

    model._set_population(torch.zeros((3, 2)))
    with pytest.raises(ValueError, match="2 columns"):
        model._append_population_rows(torch.zeros((1, 1)))

    model._append_population_rows(torch.tensor([[4.0, 0.5]]))
    assert model.population.shape == (4, 2)


@pytest.mark.smoke
def test_generic_scalar_values_are_declared_separately_from_population_fields():
    """Return core scalar results without adding them to the field registry."""
    model = Model(DEFAULT_SETTINGS.copy(), torch.device("cpu"))
    model._set_population(torch.tensor([[2.0], [4.0]]))
    model.last_born = 3
    model.last_death = 1

    values = model.get_values()

    assert set(Model.add_values()) == {"count", "avg_age", "born", "dead", "prop_aging"}
    assert values == {
        "count": 2,
        "avg_age": 3.0,
        "born": 3,
        "dead": 1,
        "prop_aging": 0.0,
    }
    assert model.population_fields == {"age": 0}


@pytest.mark.smoke
def test_generic_scalar_values_use_none_for_empty_aggregates():
    """Return null for unavailable aggregates on an empty population."""
    model = Model(make_settings(initial_population=0, initial_age_max=0), torch.device("cpu"))
    model.initialize_population()

    values = model.get_values()

    assert values["count"] == 0
    assert values["avg_age"] is None


@pytest.mark.smoke
def test_bin_values_uses_left_closed_intervals_and_overflow_counts():
    """Bin finite values and count excluded, infinite, and invalid values."""
    model = Model(make_settings(), torch.device("cpu"))
    model._set_population(torch.tensor([
        [float("-inf")], [-2.0], [-1.0], [0.0], [0.5], [1.0], [2.0],
        [float("inf")], [float("nan")],
    ]))

    result = model.bin_values("age", 2, min=-1.0, max=2.0)

    assert result == {
        "data": [2, 2],
        "min": -1.0,
        "max": 2.0,
        "below_min": 2,
        "above_max": 2,
    }


@pytest.mark.smoke
def test_bin_values_combines_padding_explicit_bounds_and_scale():
    """Use the narrowest requested range and power-scaled interval edges."""
    model = Model(make_settings(), torch.device("cpu"))
    model._set_population(torch.arange(10, dtype=torch.float32).reshape(-1, 1))

    narrowed = model.bin_values(
        "age",
        2,
        min=3.0,
        max=7.0,
        padding_min=0.2,
        padding_max=0.2,
    )
    scaled = model.bin_values("age", 2, min=0.0, max=10.0, scale=2.0)

    assert narrowed == {
        "data": [2, 2],
        "min": 3.0,
        "max": 7.0,
        "below_min": 3,
        "above_max": 3,
    }
    assert scaled["data"] == [3, 7]


@pytest.mark.smoke
def test_bin_values_handles_empty_constant_and_private_fields():
    """Return stable edge cases and reject non-public population fields."""
    empty_model = Model(make_settings(initial_population=0), torch.device("cpu"))
    empty_model.initialize_population()
    assert empty_model.bin_values("age", 3) == {
        "data": [0, 0, 0],
        "min": None,
        "max": None,
        "below_min": 0,
        "above_max": 0,
    }

    constant_model = Model(make_settings(), torch.device("cpu"))
    constant_model._set_population(torch.full((4, 1), 5.0))
    constant = constant_model.bin_values("age", 2)
    assert constant == {
        "data": [0, 4],
        "min": 4.5,
        "max": 5.5,
        "below_min": 0,
        "above_max": 0,
    }

    class PrivateFieldModel(Model):
        """Declare one field unavailable to public binning."""

        @staticmethod
        def add_population_fields():
            """Return public age and private score fields."""
            return {**Model.add_population_fields(), "score": {"public": False}}

    private_model = PrivateFieldModel(make_settings(), torch.device("cpu"))
    private_model._set_population(torch.zeros((1, 2)))
    with pytest.raises(KeyError, match="public"):
        private_model.bin_values("score", 2)


@pytest.mark.smoke
def test_generic_model_declares_no_default_batch_csv():
    """Provide an empty batch declaration when no sweep defaults exist."""
    assert Model.add_batch() == ""