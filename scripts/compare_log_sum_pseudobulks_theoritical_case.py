"""Show that the sum of the log cell type specific pseudobulks is not equal to the log of the pseudobulk."""

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from benchmark_utils import add_cell_types_grouped
from benchmark_utils.load_dataset_utils import load_cti


# %% Function to create mixed pseudobulks and plot PCA
def create_mixed_pseudobulk_transformed_pca(adata, n_samples=1000, cell_types=["Mono","CD8T"], props=[0,0.25,0.5,0.75,1], log=True):
        # Create the pseudobulks
        cell_type1 = adata[adata.obs["cell_types_grouped_2nd_level_granularity"] == cell_types[0]]
        cell_type2 = adata[adata.obs["cell_types_grouped_2nd_level_granularity"] == cell_types[1]]
        sampled_cell_type1 = np.random.choice(cell_type1.obs.index, n_samples, replace=False)
        sampled_cell_type2 = np.random.choice(cell_type2.obs.index, n_samples, replace=False)
        
        pure_cell_type1 = np.array(adata[list(sampled_cell_type1)].X.sum(axis=0).reshape(1, -1))
        pure_cell_type2 = np.array(adata[list(sampled_cell_type2)].X.sum(axis=0).reshape(1, -1))

        all_pseudobulks = []
        for prop in props:
            pseudobulk = (1-prop)*pure_cell_type1+prop*pure_cell_type2
            if log:
                pseudobulk = np.log(pseudobulk+1)
            
            all_pseudobulks.append(pseudobulk)

        all_pseudobulks = np.vstack(all_pseudobulks)
        all_pseudobulks = all_pseudobulks - np.mean(all_pseudobulks, axis=0) # center features

        # Perform PCA
        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(all_pseudobulks)

        return pca_results

# %% Run analysis
if __name__ == "__main__":
    # Load data
    adata = load_cti(n_variable_genes=100)
    adata, train_test_index = add_cell_types_grouped(adata["dataset"], "2nd_level_granularity")
    filtered_genes = adata.var.index[adata.var["highly_variable"]].tolist()
    adata = adata[:,filtered_genes]

    # %% Create mixed pseudobulks and plot PCA
    n_samples = 10000
    cell_types = ["Mono","CD8T"] # ['CD4T', 'CD8T', 'B', 'Tregs', 'Plasma', 'DC', 'NK', 'Mono', 'Mast']
    props = np.linspace(0, 1, 11)
    log=False
    pca_results = create_mixed_pseudobulk_transformed_pca(adata, n_samples=n_samples, cell_types=cell_types, props=props, log=log)

    plt.figure(figsize=(12, 8))
    for i, prop in enumerate([int(100*prop) for prop in props]):
        if prop==0:
            label = f"{'log' if log else ''} Pure cell type {cell_types[0]}"
        elif prop==100:
            label = f"{'log' if log else ''} Pure cell type {cell_types[1]}"
        else:
            label = f"{'log' if log else ''} Pseudobulk: {prop}% {cell_types[0]}, {100-prop}% {cell_types[1]}"
        plt.scatter(pca_results[i, 0], pca_results[i, 1], color=(i/len(props),i/len(props),i/len(props)),
                    label=label, marker='o', s=100, edgecolors='w')
    if not log:
        plt.ylim(-pca_results[:,0].std(), pca_results[:,0].std()) # important, otherwise it does not display at right scale
    plt.title(f"PCA of {'log' if log else ''} Mixed Pseudobulks with the Cross-tissue immune dataset")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# %%
