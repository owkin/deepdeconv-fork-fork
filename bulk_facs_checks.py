#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from scipy import integrate

from benchmark_utils import (
    load_bulk_facs,
    load_cti,
    load_bulk_facs_tpm,
    create_uniform_pseudobulk_dataset,
    create_dirichlet_pseudobulk_dataset,
    preprocess_scrna,
    add_cell_types_grouped,
)

#%%
# --- Load datasets ---
cti_data = load_cti(n_variable_genes=None)
cti_data = preprocess_scrna(cti_data["dataset"], keep_genes=None, batch_key="donor_id")
cti_data, _ = add_cell_types_grouped(cti_data, "FACS_1st_level_granularity")
single_cell_data = cti_data.to_df()

bulk_facs_data = load_bulk_facs()["dataset"].T
bulk_facs_tpm_data = load_bulk_facs_tpm()["dataset"].T

#%%
# --- Common genes across datasets ---
common_genes = list(set(bulk_facs_data.columns) & set(bulk_facs_tpm_data.columns) & set(single_cell_data.columns))

# --- Helper functions ---
def compute_variance(df, genes):
    return pd.Series(np.var(df[genes].values, axis=0, ddof=1), index=genes)

def compute_umi_stats(df, genes):
    return df[genes].sum(axis=1)

def compute_sample_variance(df, genes):
    return pd.Series(np.var(df[genes].values, axis=1, ddof=1), index=df.index)

def summarize_variance(series):
    return {"mean": series.mean(), "max": series.max(), "min": series.min()}

def summarize_umi(series):
    return {
        "mean": series.mean(),
        "max": series.max(),
        "min": series.min(),
        "median": series.median(),
    }

def calculate_cumulative_overlaps(sorted_list1, sorted_list2):
    seen_list1, seen_list2, matched = set(), set(), set()
    overlaps, n_genes = [], []
    for i in range(len(sorted_list1)):
        gene1 = sorted_list1[i]
        gene2 = sorted_list2[i]
        seen_list1.add(gene1)
        if gene1 in seen_list2:
            matched.add(gene1)
        seen_list2.add(gene2)
        if gene2 in seen_list1:
            matched.add(gene2)
        overlaps.append(len(matched))
        n_genes.append(i + 1)
    return overlaps, n_genes

# --- Plot overlap percentages ---
def percentages(overlaps, n_genes):
    return [o / n for o, n in zip(overlaps, n_genes)]

#%%
# --- Pseudobulk generation settings ---
sampling_methods = ["uniform", "dirichlet"]
cell_numbers = [10000, 300, 100]
aggregation_methods = ["sum", "mean"]

#%%
# --- Initialize data holders ---
pseudobulk_data = {}
variance_vectors = {}
umi_vectors = {}
sample_variance_vectors = {}

#%%
# --- Single-cell and Bulk calculations (cached) ---
variance_vectors["single_cell"] = compute_variance(single_cell_data, common_genes)
umi_vectors["single_cell"] = compute_umi_stats(single_cell_data, common_genes)
sample_variance_vectors["single_cell"] = compute_sample_variance(single_cell_data, common_genes)

variance_vectors["bulk_facs"] = compute_variance(bulk_facs_data, common_genes)
umi_vectors["bulk_facs"] = compute_umi_stats(bulk_facs_data, common_genes)
sample_variance_vectors["bulk_facs"] = compute_sample_variance(bulk_facs_data, common_genes)

variance_vectors["bulk_facs_tpm"] = compute_variance(bulk_facs_tpm_data, common_genes)
umi_vectors["bulk_facs_tpm"] = compute_umi_stats(bulk_facs_tpm_data, common_genes)
sample_variance_vectors["bulk_facs_tpm"] = compute_sample_variance(bulk_facs_tpm_data, common_genes)

#%%
# --- Generate pseudobulk datasets ---
total_iterations = len(sampling_methods) * len(cell_numbers) * len(aggregation_methods)
with tqdm(total=total_iterations, desc="Generating pseudobulk datasets") as pbar:
    for sampling in sampling_methods:
        for n_cells in cell_numbers:
            for aggregation in aggregation_methods:
                key = f"{sampling}_pb_{n_cells}_{aggregation}"
                creator = create_uniform_pseudobulk_dataset if sampling == "uniform" else create_dirichlet_pseudobulk_dataset
                pbar.set_description(f"Processing {key}")
                dataset = creator(
                    cti_data,
                    n_sample=206,
                    n_cells=n_cells,
                    cell_type_group="cell_types_grouped_FACS_1st_level_granularity",
                    aggregation_method=aggregation,
                )
                adata_df = dataset["adata_pseudobulk_test_counts"].to_df()
                pseudobulk_data[key] = adata_df
                variance_vectors[key] = compute_variance(adata_df, common_genes)
                umi_vectors[key] = compute_umi_stats(adata_df, common_genes)
                sample_variance_vectors[key] = compute_sample_variance(adata_df, common_genes)
                pbar.update(1)


# %%
# --- Compute final statistics ---
variance_stats = {k: summarize_variance(v) for k, v in variance_vectors.items()}
umi_stats = {k: summarize_umi(v) for k, v in umi_vectors.items()}
sample_variance_stats = {k: summarize_variance(v) for k, v in sample_variance_vectors.items()}

variance_stats_df = pd.DataFrame(variance_stats).T
umi_stats_df = pd.DataFrame(umi_stats).T
sample_variance_stats_df = pd.DataFrame(sample_variance_stats).T

# %%
# --- Output variance summary ---
print("\nVariance statistics:")
print(variance_stats_df.round(2))

# %%
# --- Output UMI summary ---
print("\nUMI statistics:")
print(umi_stats_df.round(2))

# %%
# --- Output Sample Variance summary ---
print("\nSample Variance statistics:")
print(sample_variance_stats_df.round(2))

# %% 
# --- Create boxplots to compare UMI counts across all datasets ---
plt.figure(figsize=(16, 8), dpi=300)
boxplot_data = [umi_vectors["single_cell"]] + [umi_vectors[k] for k in pseudobulk_data.keys()] + [umi_vectors["bulk_facs"], umi_vectors["bulk_facs_tpm"]]
labels = ["Single-cell"] + list(pseudobulk_data.keys()) + ["Bulk FACS", "Bulk FACS TPM"]
plt.boxplot(boxplot_data, labels=labels)
plt.yscale("log")
plt.xticks(rotation=90)
plt.ylabel("UMI Counts (log scale)")
plt.title("Distribution of UMI Counts Across Genes")
plt.tight_layout()
plt.show()

# %%
# --- Create boxplots to compare Sample Variances across all datasets ---
plt.figure(figsize=(16, 8), dpi=300)
boxplot_data = [sample_variance_vectors["single_cell"]] + [sample_variance_vectors[k] for k in pseudobulk_data.keys()] + [sample_variance_vectors["bulk_facs"], sample_variance_vectors["bulk_facs_tpm"]]
labels = ["Single-cell"] + list(pseudobulk_data.keys()) + ["Bulk FACS", "Bulk FACS TPM"]
plt.boxplot(boxplot_data, labels=labels)
plt.yscale("log")
plt.xticks(rotation=90)
plt.ylabel("Sample Variance (log scale)")
plt.title("Distribution of Sample Variances Across Genes")
plt.tight_layout()
plt.show()

# %%
# --- Overlap analysis between top variant genes ---
# Prepare sorted indices for all datasets
sorted_variances = {k: v[common_genes].sort_values(ascending=False).index for k, v in variance_vectors.items()}

# Calculate cumulative overlaps
overlap_results_bulk = {}
overlap_results_bulk_tpm = {}

for key in sorted_variances:
    if key in ["bulk_facs", "bulk_facs_tpm"]:
        continue
    overlap_bulk, _ = calculate_cumulative_overlaps(sorted_variances[key], sorted_variances["bulk_facs"])
    overlap_bulk_tpm, _ = calculate_cumulative_overlaps(sorted_variances[key], sorted_variances["bulk_facs_tpm"])
    overlap_results_bulk[key] = overlap_bulk
    overlap_results_bulk_tpm[key] = overlap_bulk_tpm

# Random overlaps
random_genes1 = common_genes.copy()
random_genes2 = common_genes.copy()
random.shuffle(random_genes1)
random.shuffle(random_genes2)
overlaps_random, n_genes_random = calculate_cumulative_overlaps(random_genes1, random_genes2)

n_genes = list(range(1, len(common_genes) + 1))

# %%
# --- Plot overlaps ---
plt.figure(figsize=(20, 20), dpi=300)
for key, overlaps in overlap_results_bulk.items():
    plt.plot(n_genes, overlaps, label=f"{key} vs Bulk", alpha=0.7)
for key, overlaps in overlap_results_bulk_tpm.items():
    plt.plot(n_genes, overlaps, label=f"{key} vs Bulk TPM", linestyle='--', alpha=0.7)
plt.plot(n_genes_random, overlaps_random, label='Random overlap', linestyle=':', alpha=0.7)
plt.plot(n_genes, n_genes, '--', label='Perfect overlap (x=y)', alpha=0.7)
plt.xlabel('Number of top variant genes considered')
plt.ylabel('Number of overlapping genes')
plt.title('Overlap with Bulk FACS and Bulk FACS TPM')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# %%
# --- Plot overlap percentages ---
plt.figure(figsize=(20, 20), dpi=300)
for key, overlaps in overlap_results_bulk.items():
    plt.plot(n_genes, percentages(overlaps, n_genes), label=f"{key} vs Bulk", alpha=0.7)
for key, overlaps in overlap_results_bulk_tpm.items():
    plt.plot(n_genes, percentages(overlaps, n_genes), label=f"{key} vs Bulk TPM", linestyle='--', alpha=0.7)
plt.plot(n_genes_random, percentages(overlaps_random, n_genes_random), label='Random overlap', linestyle=':', alpha=0.7)
plt.hlines(y=1.0, xmin=0, xmax=len(n_genes), color='g', linestyle='--', alpha=0.8, label='Perfect overlap (100%)')
plt.vlines(x=0, ymin=0, ymax=1, color='g', linestyle='--', alpha=0.8)
plt.xlabel('Number of top variant genes considered')
plt.ylabel('Percentage of overlapping genes')
plt.title('Overlap percentage with Bulk FACS and Bulk FACS TPM')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# %%
# --- AUC Calculations ---
x_norm = [x / max(n_genes) for x in n_genes]
x_norm_random = [x / max(n_genes_random) for x in n_genes_random]

print("\nArea Under Curve (AUC) Results:")
for key, overlaps in overlap_results_bulk.items():
    auc = integrate.trapz(percentages(overlaps, n_genes), x_norm)
    print(f"{key} vs Bulk FACS: {auc:.4f}")
for key, overlaps in overlap_results_bulk_tpm.items():
    auc = integrate.trapz(percentages(overlaps, n_genes), x_norm)
    print(f"{key} vs Bulk FACS TPM: {auc:.4f}")
auc_random = integrate.trapz(percentages(overlaps_random, n_genes_random), x_norm_random)
print(f"Random overlap: {auc_random:.3f}")

# %%
# --- Plot overlap percentages in 0 to 5000 range ---
plt.figure(figsize=(20, 20), dpi=300)
for key, overlaps in overlap_results_bulk.items():
    plt.plot(n_genes, percentages(overlaps, n_genes), label=f"{key} vs Bulk", alpha=0.7)
for key, overlaps in overlap_results_bulk_tpm.items():
    plt.plot(n_genes, percentages(overlaps, n_genes), label=f"{key} vs Bulk TPM", linestyle='--', alpha=0.7)
plt.plot(n_genes_random, percentages(overlaps_random, n_genes_random), label='Random overlap', linestyle=':', alpha=0.7)
plt.hlines(y=1.0, xmin=0, xmax=len(n_genes), color='g', linestyle='--', alpha=0.8, label='Perfect overlap (100%)')
plt.vlines(x=0, ymin=0, ymax=1, color='g', linestyle='--', alpha=0.8)
plt.xlim(0, 5000)
plt.ylim(0, 1)
plt.xlabel('Number of top variant genes considered')
plt.ylabel('Percentage of overlapping genes')
plt.title('Overlap percentage with Bulk FACS and Bulk FACS TPM')
plt.grid(True)
plt.legend()

# %%
# --- AUC Calculations in 0 to 5000 range ---
x_norm = [x / max(n_genes[0:5000]) for x in n_genes[0:5000]]
x_norm_random = [x / max(n_genes_random[0:5000]) for x in n_genes_random[0:5000]]

print("\nArea Under Curve (AUC) Results:")
for key, overlaps in overlap_results_bulk.items():
    auc = integrate.trapz(percentages(overlaps[0:5000], n_genes[0:5000]), x_norm)
    print(f"{key} vs Bulk FACS: {auc:.4f}")
for key, overlaps in overlap_results_bulk_tpm.items():
    auc = integrate.trapz(percentages(overlaps[0:5000], n_genes[0:5000]), x_norm)
    print(f"{key} vs Bulk FACS TPM: {auc:.4f}")
auc_random = integrate.trapz(percentages(overlaps_random[0:5000], n_genes_random[0:5000]), x_norm_random)

# %%
