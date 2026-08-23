# ChimpEvo: Evolutionary Simulation with Mutation Dynamics

## Project Description

ChimpEvo is an agent-based stochastic model that simulates the year-by-year evolution of a chimpanzee population with simplified genetics to study how emerging and inherited mutations affect lifespan.

The generic simulation lifecycle is implemented in `model.py`. The bundled beta/Gompertz mathematics lives in `model_base.py`; additional trusted `model_*.py` files can provide other models.

See [Creating a Custom Model](CREATING_MODELS.md) for the dynamic-model contract, creation steps, and a minimal example.
See [Model Catalog](MODELS.md) for the available model variants and their behavioral differences.

The application features:
- **Core simulation engine** (Python + PyTorch) for efficient population dynamics
- **Graphical interface** (Tkinter) for interactive parameter control and visualization
- **Batch processing** for parameter sweeps and multiple runs
- **Cross-platform support** (Windows, Linux, macOS with CUDA acceleration on Linux/Windows)

## Mathematical Model

### Mortality Function

The bundled `Model_base` model implements this beta/Gompertz mortality function.

Per-animal annual death probability is computed as:

$$m(t, \beta) = \alpha \cdot e^{\beta \cdot t} + \Lambda$$

Clamped to valid probability range: $m(t, \beta) \in [0, 1]$

Where:
- $m(t, \beta)$ = death probability at age $t$ for individual with genetic parameter $\beta$
- $\alpha$ = intrinsic age-related mortality multiplier
- $\beta$ = genetic parameter controlling age-dependent mortality (unbounded)
- $t$ = age in years
- $\Lambda$ = extrinsic (background) mortality rate

### Mutation Model

When offspring are produced during reproduction, one of two outcomes occurs with specified probabilities:

**With probability `mutation_probability`** (mutation occurs):
$$\beta_{new} = \frac{\beta_1 + \beta_2}{2} + \text{Uniform} \left( [-X + S \times X, X + S \times X] \right)$$

**With probability `(1 - mutation_probability)`** (no mutation):
$$\beta_{new} = \frac{\beta_1 + \beta_2}{2}$$

Where:
- $X$ = effect size of mutations (`mutation_x`): the interval half-width
- $S$ = asymmetry parameter (`mutation_s`), range $-1 \le S \le 1$; its shift is $S \times X$, not $S \times 2$
  - $S = 0$ → symmetric interval $[-X, X]$
  - $S > 0$ → shifted toward positive changes
  - $S < 0$ → shifted toward negative changes

**Example**: If $X = 3$ and $S = 0.5$:
- Asymmetry shift: $S \times X = 0.5 \times 3 = 1.5$
- Mutation interval: $[-3 + 1.5, 3 + 1.5] = [-1.5, 4.5]$
- With `mutation_probability`: draw shift $\Delta\beta = \text{Uniform}(-1.5, 4.5)$, then
  $\beta_{new} = \frac{\beta_1 + \beta_2}{2} + \Delta\beta$
- Without mutation: $\beta_{new} = \frac{\beta_1 + \beta_2}{2}$

**Note**: β values are unbounded (can be negative or arbitrarily large); extreme values affect mortality calculation.

### Population Dynamics: Year-by-Year Iteration

Each simulation year proceeds in order:

1. **Reproduction Phase**: 
   - Calculate empty niches (deaths from previous year)
   - Count mature animals (age ≥ `mature_age`)
   - Maximum annual growth limited by: mature_count × `fecundity`
   - Randomly select pairs of sexually mature animals
   - Create offspring with mutated/inherited β until reaching min(current + mature_count × fecundity, max_population)
   - New animals born with age = 0

2. **Aging Phase**: 
   - All surviving animals age by 1 year

3. **Mortality Phase**: 
   - Calculate death probability $m(t, \beta)$ for each animal
   - Stochastically remove animals based on computed probability

### Stopping Conditions

Simulation terminates when any of these occurs:

1. **Population extinction**: Fewer than 2 animals remain
2. **Maximum iterations**: 100,000 years completed
3. **Beta stabilization**: Average population β shows minimal change
   - Criterion: $|\bar{\beta}_t - \bar{\beta}_{t-1}| < \text{avg\_change}_{0-10} \times \text{stop\_beta\_change\_threshold}$
   - Where `avg_change₀₋₁₀` is the average yearly β change during first 10 years
   - Controlled by parameter `stop_beta_change_threshold` (default 0.1)

## Installation

### Prerequisites
- Python 3.9 or later
- pip package manager

### Step 1: Create Virtual Environment

```bash
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux/macOS
python -m venv .venv
source .venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: (Optional) Update PyTorch for GPU Support

If you have an NVIDIA GPU and need CUDA support optimization:

```bash
# Check CUDA version
nvcc --version

# Example: CUDA 12.4
pip uninstall torch torchvision torchaudio -y
pip cache purge
#https://pytorch.org/get-started/previous-versions/
#pip install torch --index-url https://download.pytorch.org/whl/cu124    << does not work anymore >>
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

**Note**: macOS does not support CUDA; the application will automatically use CPU.

## Usage

### 1. Graphical User Interface (Interactive)

```bash
python gui.py
```

Opens a window where you can:
- Create, switch, and confirm deletion of experiments stored under `data/<experiment>/`
- Edit all simulation parameters with real-time validation (organized in Settings tab)
- Select CPU or CUDA accelerated computation
- Start/stop simulations (automatically opens the non-modal Progress window on start)
- View live statistics, logs, and generated graphs
- Save/load configurations from JSON files

The main GUI features two tabs:
1. **Settings**: Parameter input fields, device selection, save/load config
2. **Batch**: Editable batch CSV, aggregate progress, Start/Stop controls, and result cleanup

The separate non-modal **Progress** window contains current-run tag/source details, real-time logs, performance statistics, legacy graphs, model-declared graph tabs, and calculation controls. **Stop Simulation** cancels and keeps partial output. **Finalize Simulation** completes the current simulation successfully at the end of its current year and writes final outputs. During batch execution, finalizing completes the current row and then stops the batch before the next row. Batch runs show the complete active CSV row; single runs show `Default config`. Use **Show Progress Window** to open it at any time. **Auto-scroll log** controls whether new messages move the log to its end. Closing Progress hides it without interrupting a calculation or discarding its current display.

The Settings tab contains the simulation and configuration actions and can load a selected model or all active model defaults into memory. **About Model...** opens the selected model's structured description in a modal window. Hover hints briefly explain buttons, configuration fields using their declared descriptions, and state indicators. Separate top-panel indicators show config and batch dirty state in blue. Switching experiments offers save, discard, or cancel when either editor has unsaved changes and identifies the experiment being left.

On first launch, the GUI opens a New Experiment form with an experiment-name field and a model selector populated from the discovered models. The form renders the selected model's purpose, inheritance, main rules, and differences using lightweight Markdown. The selected model supplies its default settings and optional default batch CSV. `default.conf` stores only the active experiment name. Cancelling initial creation closes the GUI without creating project data.

### 2. Single Simulation (Console)

```bash
python main.py
```

Runs one simulation using `data/<experiment>/config.json`, where `<experiment>` is selected by `default.conf`. Missing or invalid experiment selection is an error. Outputs results to `data/<experiment>/result/[tag]/`:
- `result.csv` — model-declared annual statistics
- `final.csv` — core run metadata and model-declared final values after successful completion
- `age_distribution_0000005.png` — dynamic annual graph frame with a seven-digit year suffix
- `age_distribution.png` — dynamic final graph after successful completion
- `age_distribution.gif` — animation assembled from retained numbered frames after successful completion
- The default beta model also creates corresponding `beta_distribution` and `beta_evolution` dynamic files
- `distributionN.png` — age distribution graph for each year `N` (age vs count)
- `survivorshipN.png` — smooth survivorship curve for each year `N` (log scale)
- `betaoccurrenceN.png` — beta distribution scatter plot for each year `N` when the active model declares a public `beta` field
- `distribution.gif` — animation from all `distributionN.png` frames
- `survivorship.gif` — animation from all `survivorshipN.png` frames
- `betaoccurrence.gif` — animation from all `betaoccurrenceN.png` frames when beta output is available
- `results_summary.png` — 4-subplot summary graph:
  - Population Dynamics (count over time)
  - Average Age Evolution
  - Genetic Parameter Beta Evolution  
  - Birth/Death Event Counts

### 3. Batch Processing (Parameter Sweeps)

```bash
python batch.py
```

Executes multiple simulations with parameter variants:
- The active experiment supplies `config.json`, `multi.csv`, and `result/`.
- **multi.csv** contains a required first `tag` column and may override `model` or other settings per row.
- Each row is resolved and validated before any calculation starts. Missing, incorrectly typed, or out-of-range required settings are errors.
- Results are saved to `data/<experiment>/result/[tag_from_csv]/`.
- Successful rows append their final values to authoritative `data/<experiment>/result/result.csv`; a later batch run resumes only when the tag's full resolved configuration signature is unchanged.
- Each aggregate row stores original CSV cells as `input_*`, all resolved settings, and every field from that run's `final.csv`.
- Batch tags and parameter rows must be unique. Existing aggregate rows must retain matching successful `final.csv` files and the same resolved configuration for each tag.
- GIF movies and metagraphs rebuild from active CSV tags without recalculating completed rows. Mixed-model artifacts are separated under `result/_models/<model>/`.
- The GUI Batch tab edits `multi.csv` in memory and writes it only with **Save Batch**.
- Optional batch columns can include `model`; **Delete Column** removes a selected optional column but never the required `tag` column.
- **Start Batch** runs all rows; **Run Selected Row** runs one saved row after full CSV validation. Both require saved config and batch state.
- **Stop Batch** cancels cooperatively and archives partial output. Progress **Finalize Simulation** completes and records the current row, then stops before the next row. **Clear Results** archives the complete result directory as `result_<timestamp>.bak` after confirmation.

#### Example multi.csv:

```csv
tag,mutation_probability,beta_initial
sweep_mut_0.05,0.05,0.11
sweep_mut_0.1,0.1,0.11
sweep_mut_0.2,0.2,0.11
```

## Configuration

### config.json

Experiment-local single-run configuration file with all required parameters:

*NOTE: use gui to make the config :-)

The GUI's **Load Config** accepts a JSON object and prepares it in unsaved GUI memory. **Save Config** writes the active experiment's canonical `config.json`, **Re-read Config** discards memory changes, and **Save Config As...** exports a copy. CLI and batch execution validate the persisted configuration strictly instead of adding or clamping required model values.

```json
{
  "max_population": 2000,
  "initial_population": 2000,
  "initial_age_max": 10,
  "lambda": 0.043,
  "alpha": 0.001,
  "beta_initial": 0.11,
  "mature_age": 12,
  "fecundity": 1.0,
  "mutation_probability": 0.1,
  "mutation_x": 1.0,
  "mutation_s": 0.0,
  "graph_generation_period": 1,
  "stop_beta_change_threshold": 0.1,
  "max_iterations": 100000,
  "tag": "default",
  "device": "cuda"
}
```

### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| max_population | int | 2000 | [100, 100000] | Population carrying capacity |
| initial_population | int | 2000 | [1, 100000] | Starting population size |
| initial_age_max | int | 10 | [0, 100] | Maximum initial age (sampled uniformly) |
| lambda | float | 0.043 | [0, 0.25] | Background mortality rate (Λ) |
| alpha | float | 0.001 | [0, 0.1] | Age parameter in mortality (α) |
| beta_initial | float | 0.11 | [0, 1] | Initial genetic parameter (ß) for all animals |
| mature_age | int | 12 | [1, 50] | Minimum age for sexual maturity/reproduction |
| fecundity | float | 1.0 | [0, 10] | Maximum offspring per mature animal per year (controls annual growth rate) |
| mutation_probability | float | 0.1 | [0, 0.5] | Probability that offspring undergo mutation (vs inheriting average) |
| mutation_x | float | 1.0 | [0, 10] | Effect size (X): defines mutation interval half-width; total width is 2X |
| mutation_s | float | 0.0 | [-1, 1] | Asymmetry (S): skews mutation interval toward positive/negative |
| oldest_death_percent | float | 0.1 | (0, 1] | Oldest population share used for the death-age metric |
| graph_generation_period | int | 1 | [1, 1000] | Generate yearly `distributionN/survivorshipN/betaoccurrenceN` graphs every N iterations |
| stop_beta_change_threshold | float | 0.1 | [0.01, 1.0] | Multiplier for β stabilization threshold (change < avg_first_10_years * multiplier) |
| max_iterations | int | 100000 | [100, 1000000] | Maximum simulation years before termination |
| tag | string | "default" | — | Run identifier (output directory name) |
| device | string | "cuda" | {cuda, cpu} | Compute device (auto-selects CPU if CUDA unavailable) |

## Project Structure

```
chimpevo/
├── main.py           # Simulation orchestration, graphing, CSV export
├── model.py          # Generic population-model lifecycle
├── model_base.py     # Bundled beta/Gompertz model
├── model_loader.py   # Trusted dynamic model loading
├── model_metadata.py # Dynamic model metadata validation
├── gui.py            # Tkinter graphical interface
├── batch.py          # Batch multi-run launcher
├── settings.py       # Parameter defaults and ranges
├── config.json       # Single-run configuration
├── requirements.txt  # Python dependencies
├── README.md         # This file
├── SPEC.md           # Detailed model specification
├── result/           # Output directory (created on first run)
│   └── [tag]/
│       ├── result.csv
│       ├── distribution0.png
│       ├── survivorship0.png
│       └── betaoccurrence0.png
└── .venv/            # Virtual environment (do not commit)
```

## Model Architecture

The v2 simulation separates reusable lifecycle code from model mathematics. `Model` owns the population schema, named field access, generic age behavior, scalar contracts, and binning. `Model_base(Model)` declares the bundled beta model's settings, mortality, reproduction, scalar values, graphs, metagraphs, and default batch sweep. A valid model without `beta` completes normally; beta-specific compatibility outputs are skipped.

The detailed beta method reference below applies to `Model_base`, not to every dynamic model.

### model_base.py – Bundled Beta Model

**Purpose**: Encapsulates all population dynamics calculations.

**Class: `Model_base`**

Represents the bundled population with age and genetic parameter (beta).

**State**:
- `self.population` – PyTorch tensor `[n_animals, 2]` where each row is `[age, beta]`
- `self.settings` – Configuration dictionary (parameters like alpha, lambda, etc.)
- `self.device` – torch.device (cuda or cpu)

**Constructor**:
```python
Model(settings, device)
```
Initializes empty model. Call `initialize_population()` next.

---

**Methods**:

#### `initialize_population()`
**Purpose**: Create initial population with random ages and uniform beta.

- **Input**: None. Values are read from `self.settings`.
  
- **Output**: None (modifies `self.population`)

- **Example**:
  ```python
  model.initialize_population()
  # Creates 2000 animals with ages 0–10, all with beta=0.11
  ```

---

#### `calculate_mortality_probability(ages, betas)`
**Purpose**: Compute Gompertz death probability for animals.

- **Formula**: $m(t, \beta) = \alpha \cdot e^{\beta \cdot t} + \Lambda$
  
- **Input**:
  - `ages` (torch.Tensor): Animal ages, shape `[n_animals]`
  - `betas` (torch.Tensor): Animal beta values, shape `[n_animals]`
  
- **Output**: torch.Tensor of death probabilities in [0, 1], shape `[n_animals]`

- **Implementation Detail**: Uses `torch.clamp()` to ensure valid probability range

- **Example**:
  ```python
  death_probs = model.calculate_mortality_probability(ages, betas)
  # Returns [0.02, 0.15, 0.08, ...] for each animal
  ```

---

#### `apply_mortality()`
**Purpose**: Stochastically remove animals based on death probability.

- **Algorithm**:
  1. Extract ages and betas from population
  2. Calculate death probabilities
  3. Generate random [0,1] values
  4. Mark as "survivor" if random ≥ death_prob
  5. Keep only survivors
  
- **Input**: None (uses `self.population`)

- **Output**: int = number of animals that died

- **Implementation Detail**: Uses `torch.rand_like()` for vectorized randomness

- **Example**:
  ```python
  deaths = model.apply_mortality()
  # Might return 245 if 245 animals died
  # self.population is now smaller
  ```

---

#### `mutate_beta(parent_beta1, parent_beta2)`
**Purpose**: Generate offspring beta with possible mutation.

- **Algorithm**:
  - With probability `mutation_probability`: 
    - Calculate the interval $[-X + S \times X, X + S \times X]$
    - Draw mutation shift $\Delta\beta$ from that interval
    - Add the shift to the parental average: $\beta_{new} = (\beta_1 + \beta_2)/2 + \Delta\beta$
  - With probability `1 - mutation_probability`:
    - Average the two parents: $\frac{\beta_1 + \beta_2}{2}$
  
- **Input**:
  - `parent_beta1` (float): First parent's beta
  - `parent_beta2` (float): Second parent's beta
  
- **Output**: float = offspring beta (unbounded)

- **Parameters Used**:
  - `settings["mutation_probability"]`
  - `settings["mutation_x"]` – effect size
  - `settings["mutation_s"]` – asymmetry

- **Example**:
  ```python
  child_beta = model.mutate_beta(0.10, 0.12)
  # 90% chance: child_beta = (0.10 + 0.12) / 2 = 0.11
  # For X=3 and S=0.5, S*X=1.5 and the mutation shift is Uniform(-1.5, 4.5).
  # If the sampled shift is 1.5: child_beta = 0.11 + 1.5 = 1.61
  ```

---

#### `apply_reproduction()`
**Purpose**: Breed animals to fill population up to reproduction capacity.

- **Algorithm**:
  1. Find all mature animals (age ≥ `mature_age`)
  2. If < 2 mature, return 0 (cannot breed)
  3. Calculate target population = min(current + mature_count × fecundity, max_population)
  4. While population < target:
     - Randomly select 2 mature parents (with replacement)
     - Create offspring with mutated beta
     - Add child (age 0) to population
     
- **Input**: None (uses `self.population`, settings)

- **Output**: int = number of offspring born

- **Parameters Used**:
  - `settings["max_population"]` – absolute population limit
  - `settings["mature_age"]` – minimum breeding age
  - `settings["fecundity"]` – max offspring per mature animal per year

- **Example**:
  ```python
  births = model.apply_reproduction()
  # If pop was 1950 and max is 2000:
  # Creates 50 offspring, returns 50
  # self.population is now 2000
  ```

---


### main.py – Simulation Control & Output

**Purpose**: Orchestrates simulation, collects statistics, generates graphs, exports CSV.

**Key Classes & Methods**:

- **`PopulationSimulation`** – Main simulation class
  - `__init__(settings)` – Initialize with parameters
  - `_init_population()` – Call model to set up animals
  - `_calculate_yearly_stats()` – Compute avg_age, avg_beta, count, etc. from model
  - `_should_stop()` – Check stopping conditions
  - `step()` – Execute one year: call model.apply_reproduction(), age_population(), apply_mortality()
  - `run()` – Loop step() until stopping condition
  - `_save_distribution_graph()` – Create age distribution PNG
  - `_save_survivorship_graph()` – Create survivorship curve PNG
  - `export_results()` – Write CSV, generate summary graphs, create GIFs

**Example Flow**:
```
main.py →┐
         ├→ model.apply_reproduction()  ✓ Model handles it
         ├→ model.age_population()      ✓ Model handles it
         ├→ model.apply_mortality()     ✓ Model handles it
         ├→ _calculate_yearly_stats()   ✓ Main computes stats
         ├→ _save_distribution_graph()  ✓ Main renders graph
         └→ export_results()            ✓ Main saves CSV & GIFs
```

---

### Dependency Flow

```
GUI or CLI
   ↓
main.py (PopulationSimulation)
   ├→ model.Model (population dynamics)
   │  └→ PyTorch (tensor operations, random)
   │
   ├→ Numpy (statistics, plotting helpers)
   ├→ Matplotlib (graphing)
   └→ PIL (image handling)
```

**Key Design Principle**: 
- **model.py** is pure math – no side effects, no I/O
- **main.py** is orchestration – manages model, handles output
- This separation makes model easily testable and swappable

## Development Notes

- **Code style**: Simple, readable code close to the mathematical model
- **Vectorization**: Simulation uses PyTorch tensor operations for efficiency
- **Logging**: Compatible with both console and GUI output
- **Configuration**: JSON for single runs, CSV for parameter sweeps
- **Model clarifications**: See `SPEC_.md` for detailed explanations of mutation model
- **Implementation notes**: See `memory/context.md` for design decisions and remaining TODOs
