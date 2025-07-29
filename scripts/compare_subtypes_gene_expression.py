"""Script to show that even CD4T cells and CD8T cells show variability inter- and intra- cell type gene expression."""

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from rowsetta.omics import convert
from sklearn.decomposition import PCA

from benchmark_utils import add_cell_types_grouped
from benchmark_utils.load_dataset_utils import load_cti

# %% Get cell type specific genes
def read_cell_type_specific_genes(cell_type="CD4T"):
    with open(
        f"/home/owkin/project/Simon/signature_granular_updated_corrected/DE_{cell_type}.txt",
        "r"
    ) as file:
        cell_type_genes = file.readlines()

    cell_type_genes = [line.strip().split('\t') for line in cell_type_genes]
    headers = [header.strip('"') for header in cell_type_genes[0]]
    rows = [[item.strip('"') for item in row] for row in cell_type_genes[1:]]
    cell_type_genes = pd.DataFrame(rows, columns=headers)
    return list(cell_type_genes.iloc[:,0])

# %% Run analysis
if __name__ == "__main__":
    # Load data
    adata = load_cti(n_variable_genes=100)
    adata, train_test_index = add_cell_types_grouped(adata["dataset"], "2nd_level_granularity")
    filtered_genes = adata.var.index[adata.var["highly_variable"]].tolist()
    # adata_filtered = adata[:,filtered_genes]

    # %% For "CD4T","NK", "CD8T" all together
    # Find cell type specific genes
    cell_types = ["CD4T","NK", "CD8T"]
    n_cells = 7000

    cell_type_genes = list(set(read_cell_type_specific_genes(cell_types[0]) + read_cell_type_specific_genes(cell_types[1])))
    cell_type_genes, _ = convert(cell_type_genes, input_type="hgnc", target_type="ensembl")
    cell_type_genes = adata.var_names.intersection(cell_type_genes)
    adata_filtered = adata[:,cell_type_genes]

    # %% PCA transform the gene expressions
    cell_type1 = adata_filtered[adata_filtered.obs["cell_types_grouped_2nd_level_granularity"] == cell_types[0]]
    cell_type2 = adata_filtered[adata_filtered.obs["cell_types_grouped_2nd_level_granularity"] == cell_types[1]]
    cell_type3 = adata_filtered[adata_filtered.obs["cell_types_grouped_2nd_level_granularity"] == cell_types[2]]
    sampled_cell_type1 = np.random.choice(cell_type1.obs.index, n_cells, replace=False)
    sampled_cell_type2 = np.random.choice(cell_type2.obs.index, n_cells, replace=False)
    sampled_cell_type3 = np.random.choice(cell_type3.obs.index, n_cells, replace=False)
    mean_cell_type1 = np.mean(cell_type1[sampled_cell_type1].X, axis=0)
    mean_cell_type2 = np.mean(cell_type2[sampled_cell_type2].X, axis=0)
    mean_cell_type3 = np.mean(cell_type3[sampled_cell_type3].X, axis=0)
    all_cells = np.array(np.vstack([cell_type1[sampled_cell_type1].X.toarray(),cell_type2[sampled_cell_type2].X.toarray(),cell_type3[sampled_cell_type3].X.toarray(),mean_cell_type1, mean_cell_type2,mean_cell_type3]))
    all_cells = all_cells - np.mean(all_cells, axis=0) # center features
    all_cells = all_cells[:,:40] # nitpick
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(all_cells)

    # %% Plot the results
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_results[n_cells:n_cells*2, 0], pca_results[n_cells:n_cells*2, 1], c="red", label=f"Natural Killer cells")
    plt.scatter(pca_results[:n_cells, 0], pca_results[:n_cells, 1], c="blue", label=f"CD4T cells")
    plt.scatter(pca_results[n_cells*2:n_cells*3, 0], pca_results[n_cells*2:n_cells*3, 1], c="green", alpha=0.2, label=f"CD8T cells")
    plt.scatter(pca_results[-3, 0], pca_results[-3, 1], c="blue", s=200, marker="o", edgecolors="black", linewidths=5, label=f"Mean CD4T cells")
    plt.scatter(pca_results[-2, 0], pca_results[-2, 1], c="red", s=200, marker="o", edgecolors="black", linewidths=5, label=f"Mean Natural Killer cells")
    plt.scatter(pca_results[-1, 0], pca_results[-1, 1], c="green", s=200, marker="o", edgecolors="black", linewidths=5, label=f"Mean CD8T cells")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.show()
    # %%
