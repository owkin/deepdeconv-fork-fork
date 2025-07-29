#!/bin/bash

CONFIG_DIR="benchmark_configs/split_configs_1st_level"
PYTHON_SCRIPT="run_benchmark.py"

for config in "$CONFIG_DIR"/config_test_*.yaml; do
    echo "Running benchmark for $config"
    python "$PYTHON_SCRIPT" --config "$config"
    config="$CONFIG_DIR/config_test_${n_cells}cells.yaml"
    if [ -f "$config" ]; then
        echo "Running benchmark for $config"
        python "$PYTHON_SCRIPT" --config "$config"
    else
        echo "Config file $config does not exist, skipping."
    fi
done