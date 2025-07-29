#!/bin/bash

CONFIG="benchmark_configs/config_test.yaml"
OUTDIR="benchmark_configs/split_configs_1st_level"

# Hardcoded list of n_cells values
n_cells_list=(10 25 50 75 100 125 150 200 250 300 400 500 750 1000)

for n_cells in "${n_cells_list[@]}"; do
    out_config="${OUTDIR}/config_test_${n_cells}cells.yaml"
    # Copy the original config and replace the relevant lines
    awk -v nc="$n_cells" '
        /^n_cells_per_evaluation_pseudobulk:/ {
            print "n_cells_per_evaluation_pseudobulk: [" nc "]"
            next
        }
        /^experiment_name:/ {
            print "experiment_name: \"preprint_experiments/1st_level_" nc "\" # if None, a random name will be chosen"
            next
        }
        { print }
    ' "$CONFIG" > "$out_config"
    echo "Created $out_config"
done