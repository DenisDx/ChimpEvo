# ChimpEvo: Evolutionary Simulation with Mutation Dynamics

## Project Description

ChimpEvo is an agent-based stochastic model that simulates the year-by-year evolution of a chimpanzee population with simplified genetics to study how emerging and inherited mutations affect lifespan.

!!! All the mathematics and logic of the model is implemented in the model.py module !!!

The application features:
- **Core simulation engine** (Python + PyTorch) for efficient population dynamics
- **Graphical interface** (Tkinter) for interactive parameter control and visualization
- **Batch processing** for parameter sweeps and multiple runs
- **Cross-platform support** (Windows, Linux, macOS with CUDA acceleration on Linux/Windows)

## Mathematical Model

### Mortality Function

**All the mathematics and logic of the model is implemented in the model.py module**

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
$$\beta_{new} = \text{Uniform} \left( [-X + SX, X + SX] \right)$$

**With probability `(1 - mutation_probability)`** (no mutation):
$$\beta_{new} = \frac{\beta_1 + \beta_2}{2}$$

Where:
- $X$ = effect size of mutations (`mutation_x`): larger values allow wider variation
- $S$ = asymmetry parameter (`mutation_s`), range $-1 < S < 1$
  - $S = 0$ → symmetric interval $[-X, X]$
  - $S > 0$ → shifted toward positive changes
  - $S < 0$ → shifted toward negative changes

**Example**: If $X = 2$ and $S = 0.5$:
- Mutation interval: $[-2 + 0.5 \times 2, 2 + 0.5 \times 2] = [-1, 3]$
- With `mutation_probability`: $\beta_{new} = \text{Uniform}(-1, 3)$
- Without mutation: $\beta_{new} = \frac{\beta_1 + \beta_2}{2}$

**Note**: β values are unbounded (can be negative or arbitrarily large); extreme values affect mortality calculation.

### Population Dynamics: Year-by-Year Iteration

Each simulation year proceeds in order:

1. **Reproduction Phase**: 
   - Calculate empty niches (deaths from previous year)
   - Randomly select pairs of sexually mature animals (age > `mature_age`)
   - Create offspring with mutated/inherited β until population reaches `max_population`
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
- Edit all simulation parameters with real-time validation (organized in Settings tab)
- Select CPU or CUDA accelerated computation
- Start/stop simulations (automatically switches to Progress tab on start)
- View live statistics, logs, and generated graphs
- Save/load configurations from JSON files

The GUI features three tabs:
1. **Settings**: Parameter input fields, device selection, save/load config
2. **Progress**: Real-time logs, statistics output, and live graphs (Age Distribution, Survivorship Curve, Beta Distribution)
3. **Graphs**: Auto-displays summary graphs after simulation completes

### 2. Single Simulation (Console)

```bash
python main.py
```

Runs one simulation using parameters from `config.json`. Outputs results to `result/[tag]/`:
- `result.csv` — per-year statistics (year, population count, avg age, births, deaths, avg beta)
- `distributionN.png` — age distribution graph for each year `N` (age vs count)
- `survivorshipN.png` — smooth survivorship curve for each year `N` (log scale)
- `betaoccurrenceN.png` — beta distribution scatter plot for each year `N` (beta intervals vs count, displayed as circles)
- `distribution.gif` — animation from all `distributionN.png` frames
- `survivorship.gif` — animation from all `survivorshipN.png` frames
- `betaoccurrence.gif` — animation from all `betaoccurrenceN.png` frames
- `results_summary.png` — 4-subplot summary graph:
  - Population Dynamics (count over time)
  - Average Age Evolution
  - Genetic Parameter Beta Evolution  
  - Birth/Death Event Counts

### 3. Batch Processing (Parameter Sweeps)

```bash
python batch.py [multi.csv] [config.json]
```

Executes multiple simulations with parameter variants:
- **multi.csv** contains columns matching parameter names and a `tag` column
- Each row is a separate run, inheriting unchanged parameters from base config.json
- Results are saved to `result/[tag_from_csv]/`

#### Example multi.csv:

```csv
tag,mutation_probability,beta_initial
sweep_mut_0.05,0.05,0.11
sweep_mut_0.1,0.1,0.11
sweep_mut_0.2,0.2,0.11
```

## Configuration

### config.json

Single-run configuration file with all parameters:

*NOTE: use gui to make the config :-)

```json
{
  "max_population": 2000,
  "initial_population": 2000,
  "initial_age_max": 10,
  "lambda": 0.043,
  "alpha": 0.001,
  "beta_initial": 0.11,
  "mature_age": 12,
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
| mutation_probability | float | 0.1 | [0, 0.5] | Probability that offspring undergo mutation (vs inheriting average) |
| mutation_x | float | 1.0 | [0, 10] | Effect size (X): defines mutation interval width |
| mutation_s | float | 0.0 | [-1, 1] | Asymmetry (S): skews mutation interval toward positive/negative |
| graph_generation_period | int | 1 | [1, 1000] | Generate yearly `distributionN/survivorshipN/betaoccurrenceN` graphs every N iterations |
| stop_beta_change_threshold | float | 0.1 | [0.01, 1.0] | Multiplier for β stabilization threshold (change < avg_first_10_years * multiplier) |
| max_iterations | int | 100000 | [100, 1000000] | Maximum simulation years before termination |
| tag | string | "default" | — | Run identifier (output directory name) |
| device | string | "cuda" | {cuda, cpu} | Compute device (auto-selects CPU if CUDA unavailable) |

## Project Structure

```
chimpevo/
├── main.py           # Simulation orchestration, graphing, CSV export
├── model.py          # Population dynamics mathematical model
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

The simulation is split into two modules for clean separation of concerns:

### model.py – Pure Mathematical Model

**Purpose**: Encapsulates all population dynamics calculations.

**Class: `Model`**

Represents a population of animals with age and genetic parameter (beta).

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

#### `initialize_population(initial_population, initial_age_max, beta_initial)`
**Purpose**: Create initial population with random ages and uniform beta.

- **Input**:
  - `initial_population` (int): How many animals to create
  - `initial_age_max` (int): Maximum random age (uniform 0 to max)
  - `beta_initial` (float): Initial beta value for all animals
  
- **Output**: None (modifies `self.population`)

- **Example**:
  ```python
  model.initialize_population(2000, 10, 0.11)
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
    - Draw random value from $[-X + SX, X + SX]$
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
  # 10% chance: child_beta = Uniform(-1, 3) e.g., 1.5
  ```

---

#### `apply_reproduction()`
**Purpose**: Breed animals to fill population back to carrying capacity.

- **Algorithm**:
  1. Find all mature animals (age > `mature_age`)
  2. If < 2 mature, return 0 (cannot breed)
  3. While population < `max_population`:
     - Randomly select 2 mature parents (with replacement)
     - Create offspring with mutated beta
     - Add child (age 0) to population
     
- **Input**: None (uses `self.population`, settings)

- **Output**: int = number of offspring born

- **Parameters Used**:
  - `settings["max_population"]` – target size
  - `settings["mature_age"]` – minimum breeding age

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
