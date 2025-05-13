"""Utilities to compute correlation metrics from deconvolution outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats
from loguru import logger
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .run_benchmark_constants import (
    initialize_func,
    CORRELATION_FUNCTIONS,
    GRANULARITY_TO_EVALUATION_DATASET,
    SINGLE_CELL_DATASETS,
)


def compute_benchmark_correlations(deconv_results: dict, correlation_type: str) -> pd.DataFrame:
    """General function to compute different benchmark correlations.

    Parameters
    ----------
    deconv_results: dict
        The deconvolution predictions and ground truths.
    correlation_type: str
        The correlation type to compute.

    Return
    ------
    all_results: pd.DataFrame
        The correlation results.
    """
    compute_correlation_fn, _ = initialize_func(CORRELATION_FUNCTIONS[correlation_type])
    all_results = []
    for granularity, level1 in deconv_results.items():
        evaluation_dataset = GRANULARITY_TO_EVALUATION_DATASET[granularity]
        if evaluation_dataset in SINGLE_CELL_DATASETS:
            for sampling_method, level2 in level1.items():
                for num_cells, level3 in level2.items():
                    for deconv_method, level4 in level3.items():
                        if level4["deconvolution_results"].isna().any().any():
                            # In this case, no correlation is computed
                            # TODO: allow computation on non-NaN samples
                            logger.warning(
                                f"Deconv results for the {deconv_method} method for the "
                                f"granularity {granularity} for the sampling method {sampling_method} "
                                f"for the number of cells {num_cells} contains NaN values, so "
                                "correlation computation will be skipped there."
                            )
                        else:
                            df_corr = compute_correlation_fn(level4["deconvolution_results"], level4["ground_truth"])
                            df_corr["deconv_method"] = deconv_method
                            df_corr["num_cells"] = num_cells
                            df_corr["sampling_method"] = sampling_method
                            df_corr["granularity"] = granularity
                            df_corr["correlation_type"] = correlation_type
                            all_results.append(df_corr)
        else:
            for deconv_method, level2 in level1.items():
                if level2["deconvolution_results"].isna().any().any():
                    # In this case, no correlation is computed
                    # TODO: allow computation on non-NaN samples
                    logger.warning(
                        f"Deconv results for the {deconv_method} method for the "
                        f"granularity {granularity} contains NaN values, so "
                        "correlation computation will be skipped there."
                    )

                else:
                    df_corr = compute_correlation_fn(level2["deconvolution_results"], level2["ground_truth"])
                    df_corr["deconv_method"] = deconv_method
                    df_corr["granularity"] = granularity
                    df_corr["correlation_type"] = correlation_type
                    all_results.append(df_corr)

    all_results = pd.concat(all_results, ignore_index=True)

    return all_results


def compute_correlations(deconv_results, ground_truth_fractions):
    """Compute n_sample pairwise correlations between the deconvolution results and the
    ground truth fractions of the n_groups (here n cell types).
    """
    deconv_results = deconv_results[
        ground_truth_fractions.columns
    ]  # to align order of columns
    correlations = [
        scipy.stats.pearsonr(
            ground_truth_fractions.iloc[i], deconv_results.iloc[i]
        ).statistic
        for i in range(len(deconv_results))
    ]
    correlations = pd.DataFrame({"correlations": correlations})
    correlations["sample_id"] = deconv_results.index
    return correlations


def compute_group_correlations(deconv_results, ground_truth_fractions):
    """Compute n_groups (here n cell types) pairwise correlations between the
    deconvolution results and ground truth fractions of the n_samples.
    """
    deconv_results = deconv_results[
        ground_truth_fractions.columns
    ]  # to align order of columns
    correlations = [
        scipy.stats.pearsonr(
            ground_truth_fractions.T.iloc[i], deconv_results.T.iloc[i]
        ).statistic
        for i in range(len(deconv_results.T))
    ]
    correlations = pd.DataFrame({"correlations": correlations})
    correlations["cell_types"] = deconv_results.columns
    return correlations


def compute_mse(deconv_results, ground_truth_fractions):
    """Compute Mean Squared Error between deconvolution results and ground truth fractions."""
    deconv_results = deconv_results[ground_truth_fractions.columns]  # align columns
    mse_values = [
        mean_squared_error(ground_truth_fractions.iloc[i], deconv_results.iloc[i])
        for i in range(len(deconv_results))
    ]
    mse_df = pd.DataFrame({"mse": mse_values})
    mse_df["sample_id"] = deconv_results.index
    return mse_df


def compute_mae(deconv_results, ground_truth_fractions):
    """Compute Mean Absolute Error between deconvolution results and ground truth fractions."""
    deconv_results = deconv_results[ground_truth_fractions.columns]  # align columns
    mae_values = [
        mean_absolute_error(ground_truth_fractions.iloc[i], deconv_results.iloc[i])
        for i in range(len(deconv_results))
    ]
    mae_df = pd.DataFrame({"mae": mae_values})
    mae_df["sample_id"] = deconv_results.index
    return mae_df


def concordance_correlation_coefficient(y_true, y_pred):
    """
    Calculate Lin's Concordance Correlation Coefficient.
    
    Parameters
    ----------
    y_true : array-like
        Ground truth values
    y_pred : array-like
        Predicted values
        
    Returns
    -------
    float
        Lin's Concordance Correlation Coefficient
    """
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    
    covariance = np.cov(y_true, y_pred)[0,1]
    
    # Pearson correlation coefficient
    pearson = covariance / np.sqrt(var_true * var_pred)
    
    # Bias correction factor
    C_b = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred)**2)
    
    return pearson * C_b