"""Script to show that even CD4T cells and CD8T cells show variability inter- and intra- cell type gene expression."""

# %%
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from benchmark_utils import add_cell_types_grouped
from benchmark_utils.load_dataset_utils import load_cti

# %% Function to create mixed pseudobulks and plot PCA
def transform_cell_types_pca(adata, n_cells=5000, cell_types=["CD4T","CD8T"]):
        cell_type1 = adata[adata.obs["cell_types_grouped_2nd_level_granularity"] == cell_types[0]]
        cell_type2 = adata[adata.obs["cell_types_grouped_2nd_level_granularity"] == cell_types[1]]
        sampled_cell_type1 = np.random.choice(cell_type1.obs.index, n_cells, replace=False)
        sampled_cell_type2 = np.random.choice(cell_type2.obs.index, n_cells, replace=False)
        mean_cell_type1 = np.mean(cell_type1[sampled_cell_type1].X, axis=0)
        mean_cell_type2 = np.mean(cell_type2[sampled_cell_type2].X, axis=0)
        all_cells = np.array(np.vstack([cell_type1[sampled_cell_type1].X.toarray(),cell_type2[sampled_cell_type2].X.toarray(),mean_cell_type1, mean_cell_type2]))
        all_cells = all_cells - np.mean(all_cells, axis=0) # center features
        all_cells = all_cells[:,:40] # nitpick
        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(all_cells)

        return pca_results

# %% Run analysis
if __name__ == "__main__":
    # Load data
    adata = load_cti(n_variable_genes=100)
    adata, train_test_index = add_cell_types_grouped(adata["dataset"], "2nd_level_granularity")
    filtered_genes = adata.var.index[adata.var["highly_variable"]].tolist()
    adata = adata[:,filtered_genes]

    # %% PCA transform the gene expressions
    cell_types = ["CD4T","CD8T"]
    n_cells = 5000
    pca_results = transform_cell_types_pca(adata, n_cells, cell_types)

    # %% Plot the results
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_results[n_cells:n_cells*2, 0], pca_results[n_cells:n_cells*2, 1], c="red", label=f"{cell_types[1]} cells")
    plt.scatter(pca_results[:n_cells, 0], pca_results[:n_cells, 1], c="blue", label=f"{cell_types[0]} cells")
    plt.scatter(pca_results[-2, 0], pca_results[-2, 1], c="blue", s=200, marker="o", edgecolors="black", linewidths=5, label=f"Mean {cell_types[0]} cells")
    plt.scatter(pca_results[-1, 0], pca_results[-1, 1], c="red", s=200, marker="o", edgecolors="black", linewidths=5, label=f"Mean {cell_types[1]} cells")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.title("Variability of the gene expression of two T cell subtypes")
    plt.show()
# %%
