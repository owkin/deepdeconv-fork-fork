"""Run MixUpVI dimension sweep experiments to analyze performance vs input dimension."""

import time
from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd
import scanpy as sc
from loguru import logger

import scvi
from benchmark_utils import (
    fit_mixupvi,
    preprocess_scrna,
    add_cell_types_grouped
)
from constants import (
    CAT_COV,
    CONT_COV,
    ENCODE_COVARIATES,
    TRAINING_CELL_TYPE_GROUP,
    TRAINING_DATASET,
)

# Configuration for the sweep
DIMENSION_RANGE = range(1000, 11000, 1000)  # 1k to 10k in steps of 1k
N_SEEDS = 5
SEEDS = [3, 8, 12, 23, 42]
RESULTS_DIR = Path("dimension_sweep_results")
RESULTS_DIR.mkdir(exist_ok=True)

# Metrics to track
# TODO: Change the name of the metrics below to match the actual metrics !!!!!!!!
METRICS_TO_TRACK = [
    "train_loss_epoch",
    "validation_loss_epoch",
    "train_reconstruction_loss_epoch",
    "validation_reconstruction_loss_epoch",
    "train_kl_local_epoch",
    "validation_kl_local_epoch",
    "train_mixup_loss_epoch",
    "validation_mixup_loss_epoch",
]

def run_single_experiment(adata, ranked_genes, n_genes, seed, cell_type="cell_types_grouped"):
    """Run a single experiment with given dimension and seed."""
    experiment_start = time.time()
    
    # Subset to top n_genes from ranked genes
    genes_subset = ranked_genes[:n_genes]
    adata_subset = adata[:, genes_subset].copy()
    
    # Fit model
    model = fit_mixupvi(
        adata_subset,
        model_path=None,  # Don't save individual models
        cell_type_group=cell_type,
        save_model=False,
        cat_cov=CAT_COV,
        cont_cov=CONT_COV,
        encode_covariates=ENCODE_COVARIATES,
        seed=seed,
    )
    
    # Extract metrics
    # TODO: Store the whole history of the metrics, not just the last one!!! Use the last one just for the final plot
    metrics = {metric: model.history[metric][-1] for metric in METRICS_TO_TRACK}
    metrics["training_time"] = time.time() - experiment_start
    
    return metrics

def main():
    """Run the complete dimension sweep experiment."""
    logger.info(f"Loading single-cell dataset: {TRAINING_DATASET} ...")
    
    # Load and preprocess data
    if TRAINING_DATASET == "CTI":
        adata = sc.read("/home/owkin/project/cti/cti_adata.h5ad")
        preprocess_scrna(adata, keep_genes=max(DIMENSION_RANGE), batch_key="donor_id")
        cell_type = f"cell_types_grouped_{TRAINING_CELL_TYPE_GROUP}"
    else:
        raise ValueError(f"Dataset {TRAINING_DATASET} not supported for dimension sweep")
    
    adata, train_test_index = add_cell_types_grouped(adata, TRAINING_CELL_TYPE_GROUP)
    adata_train = adata[train_test_index["Train index"]]
    ranked_genes = adata_train.var["highly_variable_rank"].sort_values().index.tolist()
    
    # Initialize results storage
    all_results = []
    
    # Track overall progress
    total_experiments = len(DIMENSION_RANGE) * N_SEEDS
    completed_experiments = 0
    
    experiment_start_time = time.time()
    
    # Run experiments
    for n_genes in DIMENSION_RANGE:
        logger.info(f"Starting experiments with {n_genes} genes...")
        for seed in SEEDS:
            logger.info(f"Running seed {seed}")
            
            try:
                metrics = run_single_experiment(adata_train, ranked_genes, n_genes, seed, cell_type)
                
                # Store results
                # TODO: Change according to the metrics and their history !!!!!!!! + change the directory names and how often to save
                result = {
                    "n_genes": n_genes,
                    "seed": seed,
                    **metrics
                }
                all_results.append(result)
                
                # Save intermediate results
                df = pd.DataFrame(all_results)
                df.to_csv(RESULTS_DIR / f"dimension_sweep_results_n_genes_{n_genes}_seed_{seed}.csv", index=False)
                
                # Update and display progress
                completed_experiments += 1
                progress = (completed_experiments / total_experiments) * 100
                elapsed_time = time.time() - experiment_start_time
                # TODO: Change the time calculation to be more accurate as it is not linear !!!!!!!!!
                avg_time_per_exp = elapsed_time / completed_experiments
                remaining_time = avg_time_per_exp * (total_experiments - completed_experiments)
                
                logger.info(
                    f"Progress: {progress:.1f}% ({completed_experiments}/{total_experiments})\n"
                    f"Average time per experiment: {avg_time_per_exp/60:.1f} minutes\n"
                    f"Estimated time remaining: {remaining_time/60:.1f} minutes"
                )
                
            except Exception as e:
                logger.error(f"Error in experiment (n_genes={n_genes}, seed={seed}): {str(e)}")
                continue
    
    # TODO: Change the directory names and how often to save
    # Save final results
    final_df = pd.DataFrame(all_results)
    final_df.to_csv(RESULTS_DIR / f"dimension_sweep_results_final.csv", index=False)
    
    # Calculate and save summary statistics
    summary_stats = final_df.groupby("n_genes").agg({
        metric: ["mean", "std"] for metric in METRICS_TO_TRACK + ["training_time"]
    }).round(4)
    
    summary_stats.to_csv(RESULTS_DIR / f"dimension_sweep_summary.csv")
    
    logger.info(f"Experiment completed! Results saved in {RESULTS_DIR}")
    logger.info(f"Total time: {(time.time() - experiment_start_time)/3600:.1f} hours")

if __name__ == "__main__":
    main() 