"""Show that the sum of the log cell type specific pseudobulks is not equal to the log of the pseudobulk."""

# %%
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from benchmark_utils import add_cell_types_grouped, load_cti


# %% Function to create mixed pseudobulks and plot PCA
def create_mixed_pseudobulk_transformed_pca(adata, n_cells=1000, cell_types=["Mono","CD8T"], props=[0,0.25,0.5,0.75,1], log=True):
        # Create the pseudobulks
        cell_type1 = adata[adata.obs["cell_types_grouped_2nd_level_granularity"] == cell_types[0]]
        cell_type2 = adata[adata.obs["cell_types_grouped_2nd_level_granularity"] == cell_types[1]]
        all_pseudobulks = []
        for prop in props:
            sampled_cell_type1 = np.random.choice(cell_type1.obs.index, int(n_cells*prop), replace=False)
            sampled_cell_type2 = np.random.choice(cell_type2.obs.index, int(n_cells*(1-prop)), replace=False)

            pseudobulk = np.array(adata[list(sampled_cell_type1)+list(sampled_cell_type2)].X.sum(axis=0).reshape(1, -1))
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
    adata = load_cti(n_variable_genes=3000)
    adata, train_test_index = add_cell_types_grouped(adata["dataset"], "2nd_level_granularity")
    filtered_genes = adata.var.index[adata.var["highly_variable"]].tolist()
    adata = adata[:,filtered_genes]

    # %% Create mixed pseudobulks and plot PCA
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for j, (n_cells, cell_types) in enumerate(zip([3000,5000],[["Mast","CD4T"],["B","Plasma"]])):
        # n_cells = 5000
        # cell_types = ["B","Plasma"] # ['CD4T', 'CD8T', 'B', 'Tregs', 'Plasma', 'DC', 'NK', 'Mono', 'Mast']
        props = np.linspace(0, 1, 100)
        log=False
        pca_results = create_mixed_pseudobulk_transformed_pca(adata, n_cells=n_cells, cell_types=cell_types, props=props, log=log)
        
        # Plot
        props_percent = [int(100 * p) for p in props]
        norm = plt.Normalize(vmin=min(props_percent), vmax=max(props_percent))
        cmap = cm.get_cmap("plasma_r")
        sc = axes[j].scatter(
            [pca_results[i, 0] for i in range(len(props))],
            [pca_results[i, 1] for i in range(len(props))],
            c=props_percent,
            cmap=cmap,
            s=100,
            edgecolors='w',
            marker='o'
        )
        cbar = plt.colorbar(sc)
        cbar.set_label(f"{'log ' if log else ''}Pseudobulk mixture % of {cell_types[0]} (or 1 - % of {cell_types[1]})")
        if log:
            coeffs = np.polyfit(pca_results[:, 0], pca_results[:, 1], deg=2)
            a, b, c = coeffs
            x_fit = np.linspace(pca_results[:, 0].min(), pca_results[:, 0].max(), 500)
            y_fit = a * x_fit**2 + b * x_fit + c
            axes[j].plot(x_fit, y_fit, color="black", lw=2)        
            axes[j].set_title(f"{cell_types[0]} and {cell_types[1]} cell types\nQuadratic coefficient: {a:.3f}")
            axes[j].set_xlim(-8,8)
            axes[j].set_ylim(-2,3)
        else:
            axes[j].set_ylim(-pca_results[:,0].std(), pca_results[:,0].std()) # important, otherwise it does not display at right scale
            axes[j].set_title(f"{cell_types[0]} and {cell_types[1]} cell types")
        axes[j].set_xlabel("PCA 1")
        axes[j].set_ylabel("PCA 2")
        axes[j].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.suptitle(f"PCA of {'log' if log else ''} Mixed Pseudobulks", fontsize=15, x=0.45)
    plt.tight_layout()
    plt.show()

# %%
