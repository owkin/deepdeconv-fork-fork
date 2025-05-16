import pandas as pd
import matplotlib.pyplot as plt
from benchmark_utils.correlation_utils import concordance_correlation_coefficient
from benchmark_utils.plotting_utils import plot_deconv_lineplot, plot_deconv_results, plot_error_metric, plot_deconv_results_and_error_metric_subplots
import os

# Load the DataFrame
# df_path = "/home/owkin/project/run_benchmark_experiments/preprint_experiments_baselines_grid/df_all_metrics.csv"
# df_path = "/home/owkin/project/run_benchmark_experiments/preprint_experiments_1st_level_100cells/df_all_metrics.csv"
# df = pd.read_csv(df_path)

granularity = "2nd_level"

# List of cell numbers to process
cell_numbers = [10, 25, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 750, 1000]

# Initialize empty list to store dataframes
dfs = []

# Load and concatenate all dataframes
for n_cells in cell_numbers:
    df_path = f"/home/owkin/project/run_benchmark_experiments/preprint_experiments/{granularity}_{n_cells}/df_all_metrics.csv"
    df = pd.read_csv(df_path)
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

# Define methods to include (excluding TAPE)
experiment_tag = f"_CTI_{granularity}_granularity"

# methods_to_include = ["Scaden", "TAPE", "NNLS", "DWLS", "OLS", "RLR", "NuSVR", "WNNLS"]
# methods_to_include = [method + experiment_tag for method in methods_to_include]
# methods_to_include.append("MixUpVI")

# Filter the DataFrame to include only selected methods
# df = df[df["deconv_method"].isin(methods_to_include)]

# Remove experiment tag from deconv_method column
df["deconv_method"] = df["deconv_method"].apply(lambda x: x.split(experiment_tag)[0] if experiment_tag in str(x) else x)

# Select specific number of cells for boxplots
n_cells = 100  # Change this to the number of cells you want to analyze
df_boxplot = df[df["num_cells"] == n_cells]

# Create both line plots and box plots for each metric
metrics = ["correlations", "mse"] #, "mae"]
# save un current folder
save_path = os.path.dirname(os.path.abspath(__file__))
# save_path = "/home/owkin/project/plots"

# Plot function mapping
plot_functions = {
    "correlations": plot_deconv_results,  # Built-in function for correlation boxplots
    "mse": lambda df, **kwargs: plot_error_metric(df, "mse", **kwargs),
    # "mae": lambda df, **kwargs: plot_error_metric(df, "mae", **kwargs),
}

df_corr = df_boxplot[df_boxplot["correlation_type"] == "sample_wise_correlation"].copy()
df_mse = df_boxplot[df_boxplot["correlation_type"] == "mse"].copy()

# Ensure all methods are present in both dataframes
all_methods = sorted(set(df_corr['deconv_method'].unique()) | set(df_mse['deconv_method'].unique()))
df_corr['deconv_method'] = pd.Categorical(df_corr['deconv_method'], categories=all_methods, ordered=True)
df_mse['deconv_method'] = pd.Categorical(df_mse['deconv_method'], categories=all_methods, ordered=True)

for metric in metrics:
    print(f"Generating plots for {metric}...")
    
    # Filter data for the current metric type
    if metric == "correlations":
        df_metric = df[df["correlation_type"] == "sample_wise_correlation"].copy()
        df_boxplot_metric = df_boxplot[df_boxplot["correlation_type"] == "sample_wise_correlation"].copy()
    else:
        df_metric = df[df["correlation_type"] == metric].copy()
        df_boxplot_metric = df_boxplot[df_boxplot["correlation_type"] == metric].copy()
    # Generate box plot
    plot_func = plot_functions[metric]
    plot_func(
        df_boxplot_metric,
        save=True,
        save_path=save_path,
        filename=f"preprint_{metric}_boxplot_{n_cells}cells"
    )
    plt.close()

# Generate line plot
plot_deconv_lineplot(
    results=df,
    metric="correlations",
    save=True,
    save_path=save_path,
    filename=f"preprint_baselines_linear_correlations_lineplot"
)
plt.close()

plot_deconv_results_and_error_metric_subplots(
    df_corr=df_corr,
    df_error=df_mse,
    error_metric="mse",
    save=True,
    save_path=save_path,
    filename=f"preprint_corr_and_mse_boxplot_{n_cells}cells"
)

print("All plots have been generated and saved!")

# Print summary statistics for the selected number of cells
# print(f"\nSummary statistics for {n_cells} cells:")
# for metric in metrics:
#     if metric == "correlations":
#         df_stats = df_boxplot[df_boxplot["correlation_type"] == "sample_wise_correlation"]
#     else:
#         df_stats = df_boxplot[df_boxplot["correlation_type"] == metric]
        
#     print(f"\n{metric.upper()} statistics:")
#     stats = df_stats.groupby("deconv_method")[metric].describe()
#     print(stats[["mean", "std", "min", "max"]])