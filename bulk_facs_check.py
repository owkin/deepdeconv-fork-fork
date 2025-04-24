# %%
from benchmark_utils import (
    load_bulk_facs, 
    load_cti,
    load_bulk_facs_tpm, 
    create_uniform_pseudobulk_dataset, 
    preprocess_scrna, 
    add_cell_types_grouped
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

# %%
# Loading single-cell CTI data
cti_data = load_cti(n_variable_genes=None)
cti_data =  preprocess_scrna(cti_data["dataset"], keep_genes=None, batch_key="donor_id")
cti_data, _ = add_cell_types_grouped(cti_data, "FACS_1st_level_granularity")
single_cell_data = cti_data.to_df()


# %%
# Generating pseudobulk dataset from single-cell CTI data
# Here is important to note that the n_cells influences the variance of the pseudobulk data ?????
# (it is the number of cells sampled to create each pseudobulk)
pseudobulk_dataset = create_uniform_pseudobulk_dataset(
    cti_data, 
    n_sample=206,
    n_cells=2000, 
    cell_type_group="cell_types_grouped_FACS_1st_level_granularity", 
    aggregation_method="sum"
    )
pseudobulk_data = pseudobulk_dataset["adata_pseudobulk_test_counts"].to_df()

# %%
# Loading bulk FACS data
bulk_facs_data = load_bulk_facs()
bulk_facs_data = bulk_facs_data["dataset"].T

# %%
# Loading bulk FACS TPM data
bulk_facs_tpm_data = load_bulk_facs_tpm()
bulk_facs_tpm_data = bulk_facs_tpm_data["dataset"].T


# %%
# Calculate variance of each gene across single-cell samples
single_cell_variances = pd.Series(
    np.var(single_cell_data.values, axis=0, ddof=1),  # ddof=1 for sample variance (same as pandas default)
    index=single_cell_data.columns
)
print(f"Variance statistics for single cell cti data ({cti_data.shape[0]} samples):")
print(f"Mean variance: {single_cell_variances.mean():.2f}")
print(f"Max variance: {single_cell_variances.max():.2f}") 
print(f"Min variance: {single_cell_variances.min():.2f}")

# %%
# Calculate variance of each gene across pseudobulk samples
pseudobulk_variances = pd.Series(
    np.var(pseudobulk_data.values, axis=0, ddof=1),
    index=pseudobulk_data.columns
)
print(f"Variance statistics for pseudobulk cti data ({pseudobulk_data.shape[0]} samples):")
print(f"Mean variance: {pseudobulk_variances.mean():.2f}")
print(f"Max variance: {pseudobulk_variances.max():.2f}") 
print(f"Min variance: {pseudobulk_variances.min():.2f}")

# %%
# Calculate variance of each gene across bulk FACS samples
bulk_facs_variances = pd.Series(
    np.var(bulk_facs_data.values, axis=0, ddof=1),
    index=bulk_facs_data.columns
)
print(f"Variance statistics for bulk_facs_data ({bulk_facs_data.shape[0]} samples):")
print(f"Mean variance: {bulk_facs_variances.mean():.2f}")
print(f"Max variance: {bulk_facs_variances.max():.2f}") 
print(f"Min variance: {bulk_facs_variances.min():.2f}")

# %%
# Calculate variance of each gene across bulk FACS TPM samples
bulk_facs_variances_tpm = pd.Series(
    np.var(bulk_facs_tpm_data.values, axis=0, ddof=1),
    index=bulk_facs_tpm_data.columns
)
print(f"Variance statistics for bulk_facs_data_tpm ({bulk_facs_tpm_data.shape[0]} samples):")
print(f"Mean variance: {bulk_facs_variances_tpm.mean():.2f}")
print(f"Max variance: {bulk_facs_variances_tpm.max():.2f}") 
print(f"Min variance: {bulk_facs_variances_tpm.min():.2f}")


# %%
# Create 2x2 grid of variance histograms
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Distribution of Gene Expression Variances Across Datasets')

# Single cell variance histogram
ax1.hist(single_cell_variances, bins=50, alpha=0.7)
ax1.set_title('Single Cell Data')
ax1.set_xlabel('Variance')
ax1.set_ylabel('Count')
ax1.set_yscale('log')

# Pseudobulk variance histogram  
ax2.hist(pseudobulk_variances, bins=50, alpha=0.7)
ax2.set_title('Pseudobulk Data')
ax2.set_xlabel('Variance')
ax2.set_ylabel('Count')
ax2.set_yscale('log')

# Bulk FACS variance histogram
ax3.hist(bulk_facs_variances, bins=50, alpha=0.7)
ax3.set_title('Bulk FACS Data')
ax3.set_xlabel('Variance')
ax3.set_ylabel('Count')
ax3.set_yscale('log')

# Bulk FACS TPM variance histogram
ax4.hist(bulk_facs_variances_tpm, bins=50, alpha=0.7)
ax4.set_title('Bulk FACS TPM Data')
ax4.set_xlabel('Variance')
ax4.set_ylabel('Count')
ax4.set_yscale('log')

plt.tight_layout()
plt.show()


# %%
# Calculate overlap between top variant genes at different thresholds
def calculate_cumulative_overlaps(sorted_list1, sorted_list2):
    """
    Calculate cumulative overlaps between two sorted lists of genes.
    
    Args:
        sorted_list1: First sorted list of gene names
        sorted_list2: Second sorted list of gene names
        
    Returns:
        overlaps: List of cumulative overlap counts
        n_genes: List of number of genes considered at each step
    """
    seen_list1, seen_list2, matched = set(), set(), set()
    overlaps, n_genes = [], []

    for i in range(len(sorted_list1)):
        gene1 = sorted_list1[i]
        gene2 = sorted_list2[i]

        # Add gene from first list and check if already seen in second list
        seen_list1.add(gene1)
        if gene1 in seen_list2:
            matched.add(gene1)

        # Add gene from second list and check if already seen in first list  
        seen_list2.add(gene2)
        if gene2 in seen_list1:
            matched.add(gene2)

        overlaps.append(len(matched))
        n_genes.append(i + 1)
        
    return overlaps, n_genes


#%%
# Get common genes between datasets (how should we do this for all the other datasets?)
common_genes = list(set(single_cell_variances.index).intersection(set(bulk_facs_variances.index)))

# Filter variances to only common genes and sort
single_cell_var_common = single_cell_variances[common_genes].sort_values(ascending=False).index
bulk_var_common = bulk_facs_variances[common_genes].sort_values(ascending=False).index

# Calculate overlaps
overlaps, n_genes = calculate_cumulative_overlaps(single_cell_var_common, bulk_var_common)


#%%
# Randomly shuffle the common genes to calculate random overlap
random_gene_order1 = common_genes.copy()
random_gene_order2 = common_genes.copy()
random.shuffle(random_gene_order1)
random.shuffle(random_gene_order2)

# Calculate overlaps
overlaps_random, n_genes_random = calculate_cumulative_overlaps(random_gene_order1, random_gene_order2)

#%%
# Plot results
plt.figure(figsize=(6,6))
plt.plot(n_genes, overlaps, label='Actual overlap')
plt.plot(n_genes_random, overlaps_random, label='Random overlap')
plt.plot(n_genes, n_genes, '--', label='Perfect overlap (x=y)', alpha=0.5)
plt.xlabel('Number of top variant genes considered')
plt.ylabel('Number of overlapping genes')
plt.title('Overlap between top variant genes in CTI and Bulk FACS\n(common genes only)')
plt.grid(True)
plt.legend()
plt.show()

print(f"\nNumber of overlapping genes in top 2000 most variant: {overlaps[2000]}")

# %%
# Plotting the overlap in terms of percentage
# Calculate percentages
percentages = [overlap/n for overlap, n in zip(overlaps, n_genes)]
percentages_random = [overlap/n for overlap, n in zip(overlaps_random, n_genes_random)]

plt.figure(figsize=(6,6))
plt.plot(n_genes, percentages, label='Actual overlap')
plt.plot(n_genes_random, percentages_random, label='Random overlap')
plt.vlines(x=0, ymin=0, ymax=1, color='g', linestyle='--', alpha=0.5)
plt.plot(n_genes, [1]*len(n_genes), '--', color='g', label='Perfect overlap (100%)', alpha=0.5)
plt.xlabel('Number of top variant genes considered')
plt.ylabel('Percentage of overlapping genes')
plt.title('Overlap percentage between top variant genes\nin CTI and Bulk FACS (common genes only)')
plt.grid(True)
plt.legend()
plt.show()

print(f"\nPercentage of overlapping genes in top 2000 most variant: {percentages[2000]:.2f}%")

# %%
plt.figure(figsize=(6,6))
plt.plot(n_genes[0:5000], percentages[0:5000], label='Actual overlap')
plt.plot(n_genes_random[0:5000], percentages_random[0:5000], label='Random overlap')
plt.plot(n_genes[0:5000], [1]*len(n_genes[0:5000]), '--', label='Perfect overlap (x=y)', alpha=0.5)
plt.xlabel('Number of top variant genes considered')
plt.ylabel('Number of overlapping genes')
plt.title('Overlap between top variant genes in CTI and Bulk FACS\n(common genes only)')
plt.grid(True)
plt.legend()
plt.show()

print(f"\nNumber of overlapping genes in top 4000 most variant: {overlaps[4000]}")
