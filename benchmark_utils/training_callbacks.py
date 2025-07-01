import lightning.pytorch as pl
from .pseudobulk_dataset_utils import create_dirichlet_pseudobulk_dataset
from .latent_signature_utils import create_latent_signature
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from scvi.model import MixUpVI
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

class LatentSpaceVisualizationCallback(pl.Callback):
    """PyTorch Lightning callback for latent space visualization during training."""

    def __init__(self, adata, cell_type_key, visualization_frequency, path_to_save_figures):
        super().__init__()
        self.cell_type_key = cell_type_key
        self.adata = adata
        self.single_cell_dataset = self._sample_cells(adata, n_cells_per_type = 40)
        self.pseudobulk_dataset = self._generate_pseudobulk_dataset(adata, n_pseudobulks = 40)
        self.visualization_frequency = visualization_frequency
        self.path_to_save_figures = path_to_save_figures
        
        # Ensure the images directory exists
        os.makedirs(f"{self.path_to_save_figures}_images", exist_ok=True)

    def _sample_cells(self, adata, n_cells_per_type):
        cell_types = adata.obs[self.cell_type_key].unique()
        sampled_indices = []
        for ct in cell_types:
            ct_indices = adata.obs[adata.obs[self.cell_type_key] == ct].index
            if len(ct_indices) >= n_cells_per_type:
                sampled = np.random.choice(ct_indices, n_cells_per_type, replace=False)
            else:
                sampled = np.random.choice(ct_indices, n_cells_per_type, replace=True)
            sampled_indices.extend(sampled)
        return adata[sampled_indices].copy()

    def _generate_pseudobulk_dataset(self, adata, n_pseudobulks = 40):
        pseudobulk_dataset = create_dirichlet_pseudobulk_dataset(adata, n_sample = n_pseudobulks, n_cells=256, cell_type_group = self.cell_type_key, aggregation_method = "mean")
        return pseudobulk_dataset["adata_pseudobulk_test_counts"]

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.visualization_frequency == 0:
            try:
                model = MixUpVI(self.adata) # TODO: This is scrappy
                model.module = pl_module.module
                model.to_device(pl_module.module.device)
                latent_sc = model.get_latent_representation(self.single_cell_dataset)
                latent_pseudobulk = model.get_latent_representation(self.pseudobulk_dataset)
                latent_signature_data = create_latent_signature(adata = self.adata, model = model, use_mixupvi = True, average_all_cells = True)

                all_latent = np.concatenate([
                    latent_sc, 
                    latent_pseudobulk, 
                    latent_signature_data.X
                ])

                labels = np.concatenate([
                    self.single_cell_dataset.obs[self.cell_type_key], 
                    np.array(["Pseudobulk"]*len(latent_pseudobulk)), 
                    [f"Signature {cell_type}" for cell_type in latent_signature_data.obs["cell type"]]
                    ])
                
                pca = PCA(n_components=2)
                tsne = TSNE(n_components=2, random_state=42)
                umap = UMAP(n_components=2, random_state=42)
                
                pca_results = pca.fit_transform(all_latent)
                tsne_results = tsne.fit_transform(all_latent)
                umap_results = umap.fit_transform(all_latent)

                unique_labels = np.unique(labels)
                colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
                color_dict = dict(zip(unique_labels, colors))
                point_colors = np.array([color_dict[label] for label in labels])

                plt.figure(figsize=(24, 6), dpi=300)

                plt.subplot(131)
                plt.scatter(pca_results[:, 0], pca_results[:, 1], c=point_colors, alpha=0.7)
                plt.title("PCA")
                plt.subplot(132)
                plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=point_colors, alpha=0.7)
                plt.title("t-SNE")
                plt.subplot(133)
                plt.scatter(umap_results[:, 0], umap_results[:, 1], c=point_colors, alpha=0.7)
                plt.title("UMAP")

                legend_elements = [plt.scatter([], [], c=[color_dict[label]], label=label) for label in unique_labels]
                plt.figlegend(handles=legend_elements, labels=list(unique_labels))

                plt.savefig(f"{self.path_to_save_figures}_images/latent_space_epoch_{trainer.current_epoch}.png")
                plt.close()
            except Exception as e:
                print(f"Warning: Failed to create latent space visualization at epoch {trainer.current_epoch}: {e}")
                # Don't let visualization errors stop training


    