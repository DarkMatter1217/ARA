from experiment_runner import run_experiment

CONFIGS = [
    "full_ara",
    "baseline_rag"   # must implement baseline manually later
]

for config in CONFIGS:
    run_experiment(run_id="run_1", model_config=config)