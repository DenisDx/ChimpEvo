# Specification Corrections

## Apply the mutation draw as a parental-average shift

**Old text (exactly)**

```text
Mutation occurs with probability mutation_probability:
New beta is randomly drawn from interval: β_new = Uniform([-X + S·X, X + S·X])
otherwise
New beta is average of parents: β_new = (β_parent1 + β_parent2) / 2
```

**Reason**

The implementation and current model contract treat the random mutation value as a shift from the parental average. The asymmetry term must be `S * X`, not `S * 2`.

**Recommended new text**

```text
Mutation occurs with probability mutation_probability:
delta_beta = Uniform([-X + S*X, X + S*X])
beta_new = (beta_parent1 + beta_parent2) / 2 + delta_beta
otherwise
beta_new = (beta_parent1 + beta_parent2) / 2
```

## Describe X as the interval half-width

**Old text (exactly)**

```text
- X (mutation_x): Effect size — defines width of mutation interval
```

**Reason**

The interval `[-X + S*X, X + S*X]` has half-width `X`, total width `2X`, and center `S*X`.

**Recommended new text**

```text
- X (mutation_x): Effect size defining the mutation interval half-width; total width is 2X
```

## Correct the mutation example semantics

**Old text (exactly)**

```text
- If mutation occurs: β_new is randomly selected from [-1, 3]
```

**Reason**

For `X = 2` and `S = 0.5`, `[-1, 3]` is the interval for the mutation shift, not the absolute offspring beta.

**Recommended new text**

```text
- If mutation occurs: delta_beta is randomly selected from [-1, 3], then beta_new = (beta_parent1 + beta_parent2) / 2 + delta_beta
```

## Align mutation parameter ranges with runtime validation

**Old text (exactly)**

```text
- mutation_x : X mutation coefficient (no range limitation; float) ; 1 by default
- mutation_s : assimetry of mutation; (-1..1); float; 0 by default
```

**Reason**

Runtime metadata validates `mutation_x` in `[0, 10]` and `mutation_s` in `[-1, 1]`. Negative `X` would also reverse the ordered uniform bounds.

**Recommended new text**

```text
- mutation_x : X mutation interval half-width; [0..10]; float; 1 by default
- mutation_s : asymmetry of mutation; [-1..1]; float; 0 by default
```