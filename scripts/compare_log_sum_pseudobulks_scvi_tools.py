"""Show that the sum of the log cell type specific pseudobulks is not equal to the log of the pseudobulk."""

# %%
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import scvi
from sklearn.decomposition import PCA

from benchmark_utils import add_cell_types_grouped, create_anndata_pseudobulk, load_cti


# %% Function to create mixed pseudobulks and plot PCA
def create_mixed_pseudobulk_transformed_pca(adata, vi_model, n_cells=1000, cell_types=["Mono","CD8T"], props=[0,0.25,0.5,0.75,1]):
        # Create the pseudobulks
        cell_type1 = adata[adata.obs["cell_types_grouped_2nd_level_granularity"] == cell_types[0]]
        cell_type2 = adata[adata.obs["cell_types_grouped_2nd_level_granularity"] == cell_types[1]]
        all_pseudobulks = []
        for prop in props:
            sampled_cell_type1 = np.random.choice(cell_type1.obs.index, int(n_cells*prop), replace=False)
            sampled_cell_type2 = np.random.choice(cell_type2.obs.index, int(n_cells*(1-prop)), replace=False)

            pseudobulk = np.array(adata[list(sampled_cell_type1)+list(sampled_cell_type2)].layers["counts"].sum(axis=0).reshape(1, -1))

            all_pseudobulks.append(pseudobulk)

        all_pseudobulks = np.vstack(all_pseudobulks)

        # Use VI model then 
        all_pseudobulks_adata = create_anndata_pseudobulk(adata_obs=adata_train.obs, adata_var_names=adata_train.var_names , x=all_pseudobulks)
        all_pseudobulks_latent = vi_model.get_latent_representation(all_pseudobulks_adata)
        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(all_pseudobulks_latent)

        return pca_results

# %% Run analysis
if __name__ == "__main__":
    # Load data
    adata = load_cti(n_variable_genes=3000)
    adata, train_test_index = add_cell_types_grouped(adata["dataset"], "2nd_level_granularity")
    adata_train = adata[train_test_index["Train index"]]
    filtered_genes = adata_train.var.index[adata_train.var["highly_variable"]].tolist()
    adata_train = adata_train[:, filtered_genes]
    # Load models
    scvi_model = scvi.model.MixUpVI.load("/home/owkin/project/Simon/plots_for_paper/mixupvi_ablation_model.pkl", adata=adata_train.copy())
    mixupvi_model = scvi.model.MixUpVI.load("/home/owkin/project/Simon/plots_for_paper/mixupvi_model.pkl", adata=adata_train.copy())

    # %% Create mixed pseudobulks and plot PCA
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for j, (n_cells, cell_types) in enumerate(zip([1000,3000],[["Mast","CD4T"],["B","Plasma"]])):
        # n_cells = 5000
        # cell_types = ["B","Plasma"] # ['CD4T', 'CD8T', 'B', 'Tregs', 'Plasma', 'DC', 'NK', 'Mono', 'Mast']
        vi_model = mixupvi_model # scvi_model, mixupvi_model
        props = np.linspace(0, 1, 100)
        pca_results = create_mixed_pseudobulk_transformed_pca(adata_train, vi_model=vi_model, n_cells=n_cells, cell_types=cell_types, props=props)
        
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
        cbar.set_label(f"Pseudobulk mixture % of {cell_types[0]} (or 1 - % of {cell_types[1]})")

        # TODO: set ylim
        # axes[j].set_ylim(-pca_results[:,0].std(), pca_results[:,0].std()) # important, otherwise it does not display at right scale
        axes[j].set_title(f"{cell_types[0]} and {cell_types[1]} cell types")
        axes[j].set_xlabel("PCA 1")
        axes[j].set_ylabel("PCA 2")
        axes[j].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.suptitle(f"PCA of Mixed Pseudobulks", fontsize=15, x=0.45)
    plt.tight_layout()
    plt.show()

# %%
