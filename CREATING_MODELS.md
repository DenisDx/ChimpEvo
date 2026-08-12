# Creating a Custom Model

ChimpEvo discovers trusted model files in the project directory. A model named
`model_example` must be stored in `model_example.py` and define a class named
`Model_example` that inherits from `Model`.

Models keep their per-animal state in `self.population`: a floating-point,
two-dimensional Torch tensor on `self.device`. Each row is one animal and each
column is declared by `add_population_fields()`.

## Model Methods

### Metadata Methods

Override these static methods to declare the model interface. The loader
validates each declaration before the simulation starts.

| Method | Required | Return value | Purpose |
|---|---|---|---|
| `add_settings()` | No | `dict` | Declare model settings. Each setting needs `description` and `default`; optional `type` is `int`, `float`, `str`, or `bool`, with optional numeric `min` and `max`. Merge `Model.add_settings()` when retaining inherited settings. |
| `add_population_fields()` | No | `dict` | Declare ordered per-animal columns as `{name: {"public": bool}}`. The `age` field is inherited and is required by `age_population()`. Public fields may be used in distribution graphs. |
| `add_values()` | No | `dict` | Declare scalar outputs returned by `get_values()`. Metadata can include `title`, `description`, `annual`, `final`, and `format`. Every declared value must be returned. |
| `add_graphs()` | No | `list[dict]` | Declare annual/final graphs. A `time` graph references declared values; a `distr` graph references public population fields. Required keys are `filename` and `values`. |
| `add_metagraphs()` | No | `list[dict]` | Declare aggregate graphs for batch runs. These reference declared values and optionally a declared setting or value through `xvalue`. |
| `add_batch()` | No | `str` | Return default batch CSV text. Return `""` when the model has no default sweep. |

### Lifecycle Methods

Override these methods to implement population dynamics. The simulation calls
them each year in this fixed order: `apply_reproduction()`, `age_population()`,
`apply_mortality()`, and `should_stop()`.

| Method | Required | Input and output | Purpose |
|---|---|---|---|
| `initialize_population()` | Usually | Uses `self.settings` and `self.device`; sets `self.population` | Create the initial population. Use `_set_population(rows)` to validate a complete tensor. |
| `apply_reproduction()` | Usually | Returns an integer | Add offspring and set `self.last_born`. Use `_append_population_rows(rows)` or assign a validated complete population tensor. |
| `age_population()` | No | No return value | Increment population age. The inherited implementation increments the declared `age` column. |
| `apply_mortality()` | Usually | Returns an integer | Remove deceased animals and set `self.last_death`. The inherited implementation keeps every animal. |
| `get_values()` | Usually | Returns `dict[str, scalar or None]` | Return all scalar values declared by `add_values()`. Values must be Python `str`, `bool`, `int`, `float`, or `None`; do not return Torch tensors. |
| `should_stop()` | No | Returns `str` or `None` | Return a readable stop reason, or `None` to continue. The engine separately enforces the maximum iteration limit. |

### Available Helpers

`Model` also provides helpers that normally do not need overriding:

- `_set_population(rows)` validates and replaces the complete population.
- `_append_population_rows(rows)` validates and appends complete rows.
- `get_population_size()` returns the current row count.
- `get_tensor(field_name)` returns one field as a detached CPU NumPy array for output.
- `get_ages()` returns the public age field.
- `bin_values(...)` produces device-side distribution bins for a public field.

Use `self.population_fields["field_name"]` instead of hard-coded column
numbers. Create tensors with `device=self.device` and a floating-point dtype so
they remain compatible with CPU and CUDA execution.

## Creation Order

1. Create `model_<name>.py` beside `model.py`.
2. Define `Model_<name>(Model)`; the class name must exactly match the file
   name after its first letter is capitalized.
3. Declare settings, fields, values, and graphs with the `add_*` methods needed
   by the model.
4. Implement initialization and the lifecycle methods that differ from `Model`.
5. Select `"model": "model_<name>"` in `config.json` or in the GUI, then run
   the smoke tests and a short simulation.

## Minimal Example

The following model keeps the inherited age-only population and reports no
births or deaths. It is a valid starting point for adding new fields and
dynamics.

```python
"""Minimal custom population model."""

import torch

from model import Model


class Model_example(Model):
    """Provide an age-only custom model."""

    @staticmethod
    def add_settings():
        """Declare inherited settings and one custom setting."""
        return {
            **Model.add_settings(),
            "starting_age": {
                "description": "Initial age for every animal",
                "default": 0,
                "type": "int",
                "min": 0,
                "max": 100,
            },
        }

    def initialize_population(self):
        """Initialize every animal with the configured starting age."""
        population_size = int(self.settings["initial_population"])
        starting_age = float(self.settings["starting_age"])
        rows = torch.full(
            (population_size, 1),
            starting_age,
            dtype=torch.float32,
            device=self.device,
        )
        self._set_population(rows)
```

Add `apply_reproduction()`, `apply_mortality()`, fields, and values as the
model acquires its own dynamics.