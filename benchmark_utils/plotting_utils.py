import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from typing import Dict
from datetime import datetime
from loguru import logger


def plot_benchmark_correlations(df_all_correlations, save_path: str = "", save: bool = True):
    """General function to plot benchmark correlations and error metrics, and save them by default."""
    def _get_groups(df, groupby_col):
        """Returns grouped DataFrames if groupby_col exists and is not empty, else returns a list with the original df."""
        if groupby_col in df.columns and df[groupby_col].notna().any():
            return [group for _, group in df.groupby(groupby_col)]
        return [df]
    
    plot_func_map = {
        "sample_wise_correlation": plot_deconv_results,
        "cell_type_wise_correlation": plot_deconv_results_group,
        "sample_wise_mse": lambda df, **kwargs: plot_error_metric(df, "mse", **kwargs),
        "sample_wise_mae": lambda df, **kwargs: plot_error_metric(df, "mae", **kwargs)
    }

    for granularity in df_all_correlations.granularity.unique():
        df_all_correlations_temp = df_all_correlations[df_all_correlations["granularity"] == granularity]
        if "num_cells" in df_all_correlations_temp.columns and df_all_correlations_temp.num_cells.dropna().nunique() > 1:
            # Multiple num_cells were computed
            df_to_plot = df_all_correlations_temp[df_all_correlations_temp["correlation_type"] == "sample_wise_correlation"]
            for group in _get_groups(df_to_plot, "sampling_method"):
                plot_deconv_lineplot(group, metric="correlations", save=save, save_path=save_path)
        else:
            # One pseudobulk num_cells or bulk
            for correlation_type in df_all_correlations_temp["correlation_type"].unique():
                df_to_plot = df_all_correlations_temp[df_all_correlations_temp["correlation_type"] == correlation_type]
                plot_func = plot_func_map.get(correlation_type, plot_deconv_results)
                for group in _get_groups(df_to_plot, "sampling_method"):
                    plot_func(group, save=save, save_path=save_path)


def plot_purified_deconv_results(deconv_results, only_fit_one_baseline, more_details=False, save=False, filename="test"):
    """Plot the deconv results from sanity check 1"""
    if not more_details:
        if only_fit_one_baseline:
            deconv_results = deconv_results.loc[
                deconv_results["Cell type predicted"] == deconv_results["Cell type"]
            ].copy()
            deconv_results["Method"] = "NNLS"
        hue = "Method"
    else:
        hue = "Cell type predicted"

    plt.clf()
    sns.set_style("whitegrid")
    sns.stripplot(
        data=deconv_results, x="Cell type", y="Estimated Fraction", hue=hue
    )
    plt.show()
    if save:
        plt.savefig(f"/home/owkin/project/plots/{filename}.png", dpi=300)


def plot_deconv_results(correlations, save=False, save_path="", filename=""):
    """Plot the deconv correlation results from sanity checks 2 and 3."""
    if filename == "":
        granularity = correlations["granularity"].unique()[0]
        if "sampling_method" in correlations.columns and isinstance(correlations["sampling_method"].unique()[0],str):
            sampling_method = correlations["sampling_method"].unique()[0]
            filename = f"{granularity}_{sampling_method}_sampling_correlation_boxplot"
        else:
            filename = f"{granularity}_correlation_boxplot"

    correlations = correlations[["correlations","deconv_method"]]
    plt.clf()
    sns.set_style("whitegrid")
    # Boxplot with wider boxes, smaller outliers, and lighter palette
    plt.figure(figsize=(10, 6))
    boxplot = sns.boxplot(
        data=correlations,
        x="deconv_method",
        y="correlations",
        hue="deconv_method",
        width=0.8,  # Wider boxes,
        dodge=False,
        palette=sns.color_palette("pastel"),  # Lighter palette
        flierprops=dict(marker='o', markersize=2, linestyle='none', markerfacecolor='gray')  # Smaller outliers
    )
    plt.xticks(rotation=90)
    # Plot the medians
    x_categories = [t.get_text() for t in boxplot.get_xticklabels()] # order of categories
    medians = (
        correlations.groupby("deconv_method")["correlations"]
        .median()
        .reindex(x_categories)  # ensure same order as x-axis
        .round(4)
    )
    y_range = correlations["correlations"].max() - correlations["correlations"].min()
    vertical_offset = y_range * 0.0005
    y_position = medians + vertical_offset
    for xtick, method in enumerate(x_categories):
        median_value = medians.loc[method]
        if np.isfinite(median_value):
            boxplot.text(
                xtick,
                y_position.loc[method],
                f"{median_value:.3f}",
                color="black",
                ha="center",
                weight="semibold",
                fontsize=10,
            )
    # Remove the title (do not set plt.title())
    if save:
        plt.savefig(f"{save_path}/{filename}.png", dpi=300, bbox_inches='tight')
        plt.close()

def plot_deconv_results_group(correlations_group, save=False, save_path="", filename=""):
    """Plot the deconv correlation results from sanity checks 2 and 3.
    per cell type.
    """
    if filename == "":
        granularity = correlations_group["granularity"].unique()[0]
        if "sampling_method" in correlations_group.columns and isinstance(correlations_group["sampling_method"].unique()[0], str):
            sampling_method = correlations_group["sampling_method"].unique()[0]
            filename = f"{granularity}_{sampling_method}_sampling_cell_type_correlation_plot"
        else:
            filename = f"{granularity}_cell_type_plot"

    df = correlations_group[["correlations","deconv_method", "cell_types"]]
    df = df.fillna(0) # Replace NaN with zeros
    plt.clf()
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(x="cell_types", y="correlations", hue="deconv_method", data=df)
    plt.legend()
    plt.xlabel("Cell Type")
    plt.ylabel("Correlation")
    plt.title("Bar Plot of Correlations by Cell Type and Model")
    plt.show()
    if save:
        plt.savefig(f"{save_path}/{filename}.png", dpi=300)


def plot_error_metric(correlations, metric_name, save=False, save_path="", filename=""):
    """Plot error metrics (MSE or MAE) boxplots."""
    if filename == "":
        granularity = correlations["granularity"].unique()[0]
        if "sampling_method" in correlations.columns and isinstance(correlations["sampling_method"].unique()[0],str):
            sampling_method = correlations["sampling_method"].unique()[0]
            filename = f"{granularity}_{sampling_method}_sampling_{metric_name}_boxplot"
        else:
            filename = f"{granularity}_{metric_name}_boxplot"

    correlations = correlations[[metric_name, "deconv_method"]]
    plt.clf()
    sns.set_style("whitegrid")
    # Boxplot
    plt.figure(figsize=(10, 6))
    boxplot = sns.boxplot(
        data=correlations,
        x="deconv_method",
        y=metric_name,
        hue="deconv_method",
        width=0.8,  # Wider boxes
        dodge=False,
        palette=sns.color_palette("pastel"),  # Lighter palette
        flierprops=dict(marker='o', markersize=2, linestyle='none', markerfacecolor='gray')  # Smaller outliers
    )
    plt.xticks(rotation=90)
    
    # Plot the medians
    x_categories = [t.get_text() for t in boxplot.get_xticklabels()]
    medians = (
        correlations.groupby("deconv_method")[metric_name]
        .median()
        .reindex(x_categories)
        .round(4)
    )
    y_range = correlations[metric_name].max() - correlations[metric_name].min()
    vertical_offset = y_range * 0.0005
    y_position = medians + vertical_offset
    
    for xtick, method in enumerate(x_categories):
        median_value = medians.loc[method]
        if np.isfinite(median_value):
            boxplot.text(
                xtick,
                y_position.loc[method],
                f"{median_value:.4f}",
                color="black",
                ha="center",
                weight="semibold",
                fontsize=10,
            )
    
    # plt.title(f"{metric_name.upper()} by Deconvolution Method")
    if save:
        plt.savefig(f"{save_path}/{filename}.png", dpi=300, bbox_inches='tight')
        plt.close()


def plot_deconv_lineplot(results: Dict[int, pd.DataFrame],
                         metric: str = "correlations",
                         save: bool = False,
                         save_path: str = "",
                         filename: str = ""):
    """Plot metrics vs number of cells as a line plot."""
    if filename == "":
        granularity = results["granularity"].unique()[0]
        sampling_method = results["sampling_method"].unique()[0]
        filename = f"{granularity}_{sampling_method}_sampling_numcells_{metric}_lineplot"

    # Configure plot settings based on metric
    metric_settings = {
        "correlations": {
            "ylabel": "Correlation"
        },
        "mse": {
            "ylabel": "MSE"
        },
        "mae": {
            "ylabel": "MAE"
        }
    }

    if metric not in metric_settings:
        logger.warning(f"Unknown metric {metric}, using default correlation settings")
        metric_settings[metric] = metric_settings["correlations"]

    # Ensure we have the required columns
    required_cols = [metric, "deconv_method", "num_cells"]
    missing_cols = [col for col in required_cols if col not in results.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}. Available columns: {results.columns}")

    plt.clf()
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Create line plot
    sns.lineplot(
        data=results,
        x="num_cells",
        y=metric,
        hue="deconv_method"
    )

    # Set labels
    plt.xlabel("Number of cells per pseudobulk")
    plt.ylabel(metric_settings[metric]["ylabel"])
    
    # Add legend with better positioning
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    if save:
        path = f"{save_path}/{filename}.png"
        if os.path.isfile(path):
            new_path = f"{save_path}/{filename}_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.png"
            logger.warning(f"{path} already exists. Saving file on this path instead: {new_path}")
            path = new_path
        plt.savefig(path, dpi=300, bbox_inches='tight')
        logger.debug(f"Plot saved to the following path: {path}")
    
    plt.show()


#### LOSSES PLOTS

def plot_metrics(model_history, train: bool = True, n_epochs: int = 100):
    """Plot the train or val metrics from training."""
    if train:
        suffix = "train"
    else:
        suffix = "validation"
    plt.clf()
    plt.plot(
        range(n_epochs),
        model_history[f"pearson_coeff_{suffix}"],
        label="Latent space Pearson coefficient",
    )
    plt.plot(
        range(n_epochs),
        model_history[f"pearson_coeff_deconv_{suffix}"],
        label="Deconv pearson coefficient",
    )
    plt.plot(
        range(n_epochs),
        model_history[f"cosine_similarity_{suffix}"],
        label="Deconv cosine similarity",
    )
    plt.legend()
    plt.title(f"{suffix} metrics")
    plt.show()

def plot_mse_mae_deconv(model_history, train: bool = True, n_epochs: int = 100):
    """Plot the train or val MSE and MAE deconv errors from training."""
    if train:
        suffix = "train"
    else:
        suffix = "validation"
    plt.clf()
    plt.plot(
        range(n_epochs), 
        model_history[f"mse_deconv_{suffix}"], 
        label="Deconv MSE error",
    )
    plt.plot(
        range(n_epochs), 
        model_history[f"mae_deconv_{suffix}"], 
        label="Deconv MAE error",
    )
    plt.legend()
    plt.title(f"{suffix} errors")
    plt.show()

def plot_loss(model_history, n_epochs: int = 100):
    """Plot the train and val loss from training."""
    plt.clf()
    plt.plot(range(n_epochs), model_history["train_loss_epoch"], label="Train")
    plt.plot(
        range(n_epochs),
        model_history["validation_loss"],
        label="Validation",
    )
    plt.legend()
    plt.title("Loss epochs")
    plt.show()

def plot_mixup_loss(model_history, n_epochs: int = 100):
    """Plot the train and val mixup loss from training."""
    plt.clf()
    plt.plot(range(n_epochs), model_history["mixup_penalty_train"], label="Train")
    plt.plot(
        range(n_epochs),
        model_history["mixup_penalty_validation"],
        label="Validation",
    )
    plt.legend()
    plt.title("Mixup loss epochs")
    plt.show()

def plot_reconstruction_loss(model_history, n_epochs: int = 100):
    """Plot the train and val reconstruction loss from training."""
    plt.clf()
    plt.plot(range(n_epochs), model_history["reconstruction_loss_train"], label="Train")
    plt.plot(
        range(n_epochs),
        model_history["reconstruction_loss_validation"],
        label="Validation",
    )
    plt.legend()
    plt.title("Reconstruction loss epochs")
    plt.show()

def plot_kl_loss(model_history, n_epochs: int = 100):
    """Plot the train and val KL loss from training."""
    plt.clf()
    plt.plot(range(n_epochs), model_history["kl_local_train"], label="Train")
    plt.plot(
        range(n_epochs),
        model_history["kl_local_validation"],
        label="Validation",
    )
    plt.legend()
    plt.title("KL loss epochs")
    plt.show()

def plot_pearson_random(model_history, train: bool = True, n_epochs: int = 100):
    """Plot the train or val random vs normal pearson deconv metrics from training."""
    if train:
        suffix = "train"
    else:
        suffix = "validation"
    plt.clf()
    plt.plot(
        range(n_epochs),
        model_history[f"pearson_coeff_deconv_{suffix}"],
        label="Deconv pearson coefficient",
    )
    plt.plot(
        range(n_epochs),
        model_history[f"pearson_coeff_random_{suffix}"],
        label="Deconv pearson coefficient random vector",
    )
    plt.legend()
    plt.title(f"{suffix} metrics")
    plt.show()

def compare_tuning_results(
      all_results, variable_to_plot: str, variable_tuned: str, n_epochs: int = 100, hp_index_to_plot: list = None
):
    """Plot the train or val losses for a selection of hyperparameters."""
    all_hp = all_results[variable_tuned].unique()
    if all_hp.dtype != "0":
        all_hp.sort()
    if hp_index_to_plot is not None:
        hp_to_plot = all_hp[hp_index_to_plot]
        all_results = all_results.loc[all_results[variable_tuned].isin(hp_to_plot)]
    
    custom_palette = sns.color_palette("husl", n_colors=len(all_results[variable_tuned].unique()))
    all_results["epoch"] = all_results.index
    if (n_nan := all_results[variable_to_plot].isna().sum()) > 0:
        print(
            f"There are {n_nan} missing values in the variable to plot ({variable_to_plot})."
            "Filling them with the next row values."
        )
        all_results[variable_to_plot] = all_results[variable_to_plot].fillna(method='bfill')
    sns.set_theme(style="darkgrid")
    sns.lineplot(x="epoch", y=variable_to_plot, hue=variable_tuned, ci="sd", data=all_results, err_style="bars", palette=custom_palette)
    plt.show()

## SUBPLOTS
def plot_deconv_results_and_error_metric_subplots(
    df_corr, df_error, error_metric,
    save=False, save_path="", filename=""
):
    """
    Plot correlation and error metric (MSE or MAE) boxplots side by side with a shared legend.
    """
    # Prepare data
    corr_data = df_corr[["correlations", "deconv_method"]]
    error_data = df_error[[error_metric, "deconv_method"]]

    # Set up the figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=False)

    # Common palette
    palette = sns.color_palette("pastel")

    # Boxplot for correlations
    box1 = sns.boxplot(
        data=corr_data,
        x="deconv_method",
        y="correlations",
        hue="deconv_method",
        width=0.8,
        palette=palette,
        dodge=False,
        flierprops=dict(marker='o', markersize=2, linestyle='none', markerfacecolor='gray'),
        ax=axes[0]
    )
    axes[0].set_xlabel("Deconvolution Method")
    axes[0].set_ylabel("Correlation")
    axes[0].set_title("Correlation", fontsize=16)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90, fontsize=13)

    # Plot the medians for correlations
    x_categories = [t.get_text() for t in axes[0].get_xticklabels()]
    medians = (
        corr_data.groupby("deconv_method")["correlations"]
        .median()
        .reindex(x_categories)
        .round(4)
    )
    y_range = corr_data["correlations"].max() - corr_data["correlations"].min()
    vertical_offset = y_range * 0.0005
    y_position = medians + vertical_offset
    for xtick, method in enumerate(x_categories):
        median_value = medians.loc[method]
        if np.isfinite(median_value):
            axes[0].text(
                xtick,
                y_position.loc[method],
                f"{median_value:.3f}",
                color="black",
                ha="center",
                weight="semibold",
                fontsize=10,
            )

    # Boxplot for error metric
    box2 = sns.boxplot(
        data=error_data,
        x="deconv_method",
        y=error_metric,
        hue="deconv_method",
        width=0.8,
        dodge=False,
        palette=palette,
        flierprops=dict(marker='o', markersize=2, linestyle='none', markerfacecolor='gray'),
        ax=axes[1]
    )
    axes[1].set_xlabel("Deconvolution Method")
    axes[1].set_ylabel(error_metric.upper())
    axes[1].set_title(error_metric.upper(), fontsize=16)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=90, fontsize=13)

    # Plot the medians for error metric
    x_categories_err = [t.get_text() for t in axes[1].get_xticklabels()]
    medians_err = (
        error_data.groupby("deconv_method")[error_metric]
        .median()
        .reindex(x_categories_err)
        .round(4)
    )
    y_range_err = error_data[error_metric].max() - error_data[error_metric].min()
    vertical_offset_err = y_range_err * 0.0005
    y_position_err = medians_err + vertical_offset_err
    for xtick, method in enumerate(x_categories_err):
        median_value = medians_err.loc[method]
        if np.isfinite(median_value):
            axes[1].text(
                xtick,
                y_position_err.loc[method],
                f"{median_value:.4f}",
                color="black",
                ha="center",
                weight="semibold",
                fontsize=10,
            )

    # Remove individual legends
    axes[0].legend_.remove()
    axes[1].legend_.remove()

    # Add a single shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), bbox_to_anchor=(0.5, 1.08), fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.98])  # leave space for legend

    if save:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(f"{save_path}/{filename}.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

