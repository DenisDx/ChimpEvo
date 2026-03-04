"""
Batch simulation runner: execute multiple simulations with parameter sweeps
Loads parameter variations from multi.csv and runs sequentially
"""

import csv
import json
from pathlib import Path
from main import run_simulation, log
from settings import DEFAULT_SETTINGS


def run_batch(csv_path="multi.csv", config_path="config.json"):
    """Run multiple simulations with parameter variants
    
    Args:
        csv_path: multi.csv file with parameter variants (tag in first column)
        config_path: base config.json to merge with variants
        
    Returns:
        list of result directories
    """
    # Load base configuration
    if Path(config_path).exists():
        with open(config_path) as f:
            base_config = json.load(f)
    else:
        base_config = DEFAULT_SETTINGS.copy()
    
    # Load variants from CSV
    variants = []
    if Path(csv_path).exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                variants.append(row)
    else:
        log(f"Warning: {csv_path} not found, using base config only")
        variants = [{"tag": base_config.get("tag", "default")}]
    
    results_dirs = []
    
    for i, variant in enumerate(variants):
        log(f"\n{'='*60}")
        log(f"Batch run {i+1}/{len(variants)}")
        log(f"{'='*60}")
        
        # Merge variant with base config
        config = base_config.copy()
        
        tag = None
        for key, val_str in variant.items():
            if key == "tag":
                tag = val_str
                config["tag"] = tag
                continue
            
            if val_str is None or val_str == "":
                continue
            
            # Validate parameter exists
            if key not in DEFAULT_SETTINGS and key != "tag":
                log(f"Warning: unknown parameter '{key}' in variant, skipping")
                continue
            
            # Parse value
            try:
                # Infer type from default setting
                if key in DEFAULT_SETTINGS:
                    default = DEFAULT_SETTINGS[key]
                    if isinstance(default, bool):
                        config[key] = val_str.lower() in ["true", "1", "yes"]
                    elif isinstance(default, int):
                        config[key] = int(val_str)
                    else:
                        config[key] = float(val_str)
                else:
                    config[key] = val_str
            except ValueError:
                log(f"Error parsing {key}={val_str}, using default")
                continue
        
        log(f"Running with tag '{tag or 'default'}'")
        log(f"Config: {json.dumps(config, indent=2)[:200]}...")
        
        # Run simulation
        try:
            run_simulation(config)
            results_dirs.append(f"result/{config.get('tag', 'default')}")
        except Exception as e:
            log(f"Error in batch run {i+1}: {e}")
    
    log(f"\n{'='*60}")
    log(f"Batch complete: {len(results_dirs)} simulations finished")
    log(f"Results in: {results_dirs}")
    
    return results_dirs


if __name__ == "__main__":
    import sys
    
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "multi.csv"
    config_file = sys.argv[2] if len(sys.argv) > 2 else "config.json"
    
    run_batch(csv_file, config_file)
