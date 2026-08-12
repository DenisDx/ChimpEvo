# Model Catalog

For each model, record its purpose, key new settings, differences from related
models, and any other brief operational notes.

## model_base

- **Purpose:** Baseline beta/Gompertz population model.
- **Key settings:** Population, mortality, mutation, maturity, and fecundity settings.
- **Difference:** Reproduction samples mature parents with replacement; one animal may participate in any number of births during a year.

## model_base_fast

- **Purpose:** Speed up `model_base` reproduction.
- **Key settings:** None beyond `model_base`.
- **Difference:** Batches parent selection, mutation, child creation, and population concatenation on the selected Torch device. It keeps the unlimited per-parent annual reproduction behavior.

## model_base_fast_fixed_fecundity

- **Purpose:** Limit each mature animal's annual reproductive participation while retaining batched reproduction.
- **Key settings:** `fecundity` is an integer per-parent annual participation limit; its fractional part is ignored.
- **Difference:** Each offspring consumes one available slot from each of two parents. A parent appears at most `floor(fecundity)` times; maximum births are half the available parent slots. Sexes are still not modeled.
