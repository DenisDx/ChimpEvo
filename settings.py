# Default simulation parameters
DEFAULT_SETTINGS = {
    "max_population": 2000,
    "initial_population": 2000,
    "initial_age_max": 10,
    "lambda": 0.043,  # extrinsic mortality (Λ)
    "alpha": 0.001,   # age-related mortality multiplier (α)
    "beta_initial": 0.11,  # initial genetic parameter (ß)
    "mature_age": 12,  # minimum age for reproduction
    "mutation_probability": 0.1,  # probability of mutation per reproduction
    "mutation_x": 1.0,  # effect size of mutations (X)
    "mutation_s": 0.0,  # asymmetry of mutations (S), range [-1, 1]
    "stat_generation_period": 1,  # collect statistics every N iterations (performance optimization)
    "graph_generation_period": 1,  # generate yearly graphs every N stat collections
    "stop_beta_change_threshold": 0.1,  # multiplier for beta stabilization threshold
    "max_iterations": 100000,  # maximum simulation iterations before stopping
    "tag": "default",  # run identifier
    "device": "cuda",  # "cuda" or "cpu"
}

# Validation ranges for GUI
PARAMETER_RANGES = {
    "max_population": (100, 100000000),
    "initial_population": (1, 100000000),
    "initial_age_max": (0, 100),
    "lambda": (0.0, 0.25),
    "alpha": (0.0, 0.1),
    "beta_initial": (0.0, 1.0),
    "mature_age": (1, 50),
    "mutation_probability": (0.0, 0.5),
    "mutation_x": (0.0, 10.0),
    "mutation_s": (-1.0, 1.0),
    "stat_generation_period": (1, 10000),
    "graph_generation_period": (1, 1000),
    "stop_beta_change_threshold": (0.0001, 1.0),
    "max_iterations": (100, 1000000),
}

# Parameter descriptions for GUI
PARAMETER_DESCRIPTIONS = {
    "max_population": "Population carrying capacity",
    "initial_population": "Starting population count",
    "initial_age_max": "Max random initial age",
    "lambda": "Background mortality rate (Λ)",
    "alpha": "Age-related mortality (α)",
    "beta_initial": "Initial genetic parameter (β)",
    "mature_age": "Min age for reproduction",
    "mutation_probability": "Mutation chance per birth",
    "mutation_x": "Mutation effect size (X)",
    "mutation_s": "Mutation asymmetry (S)",
    "stat_generation_period": "Collect stats every N years",
    "graph_generation_period": "Generate graphs every N stats",
    "stop_beta_change_threshold": "Beta stabilization multiplier",
    "max_iterations": "Maximum simulation years",
    "device": "Compute device (CUDA or CPU)",
}

# Parameter groups for GUI organization
PARAMETER_GROUPS = {
    "Model Parameters": [
        "max_population",
        "initial_population", 
        "initial_age_max",
        "lambda",
        "alpha",
        "beta_initial",
        "mature_age",
        "mutation_probability",
        "mutation_x",
        "mutation_s",
    ],
    "Performance & Output": [
        "stat_generation_period",
        "graph_generation_period",
        "stop_beta_change_threshold",
        "max_iterations",
    ]
}
