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

## model_base_diploid

- **Purpose:** Add diploid beta inheritance to `model_base_z` while preserving fixed parent fecundity, Z-biased mutations, and existing mortality/output behavior.
- **Key settings:** No new settings. The model inherits `mutation_probability`, $X$, $S$, and $Z$.
- **Inheritance:** A child independently receives one random allele (`beta1` or `beta2`) from each parent. Each inherited allele has its own mutation-probability check and, when selected, its own independently sampled X/S/Z shift.
- **Codominance:** The stored phenotype is $beta=(beta1+beta2)/2$. Both allele sets contribute equally because beta represents the aggregate behavior of many genes rather than dominance at a single gene.
- **Compatibility:** Population rows keep public `beta` at column 1 and store private `beta1` and `beta2` after it. Mortality, beta statistics, graphs, and stabilization remain inherited and consume the stored phenotype.

## model_alleles

- **Purpose:** Simulate `N_alleles` independent diploid beta loci with optional dominance and allele-specific delayed age effects.
- **Key settings:** `N_alleles`, `delta_x`, `delta_reversion`, and `use_dominance`; inherited `mutation_probability`, $X$, $S$, $Z$, and `beta_only_positive` remain available in batch configurations.
- **Inheritance:** Every child independently takes one allele at each locus from each parent. One mutation check per inherited allele jointly updates beta and its optional dominance/delta attributes.
- **Phenotype:** Without delta, public beta is the mean of all alleles or the selected dominant allele per pair. With delta, the model recalculates an effective beta before mortality from $beta_i(t-delta_i)/t$. This is an effective-beta approximation, not an average of per-allele mortality probabilities.
- **Compatibility:** The public `age` and `beta` fields remain first; all dynamic allele fields are private and are not written as CSV columns.
