#!/bin/bash

CONFIG_DIR="benchmark_configs/split_configs_2nd_level"
PYTHON_SCRIPT="run_benchmark.py"

for config in "$CONFIG_DIR"/config_test_*.yaml; do
    echo "Running benchmark for $config"
    python "$PYTHON_SCRIPT" --config "$config"
done