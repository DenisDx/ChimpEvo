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

## model_base_z

- **Purpose:** Study an independent bias in the probability of positive versus negative beta mutations while retaining fixed per-parent fecundity and batched reproduction.
- **Key settings:** `mutation_z` is the sign bias $Z$ in $[-1, 1]$. The positive branch is selected with probability $(Z+1)/2$.
- **Difference:** After mutation occurs, the model samples either $U(0, X(S+1))$ or $U(X(S-1), 0)$. $Z$ controls branch probability, while $S$ controls the two interval lengths.
- **Equivalence:** When $Z=S$, the mixture is distributionally equivalent to the parent model's $U(X(S-1), X(S+1))$ mutation shift. Exact seeded trajectories may differ because branch selection consumes an additional random draw.
