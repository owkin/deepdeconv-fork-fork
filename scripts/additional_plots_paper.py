"""A few quick plots done for the paper."""

# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scanpy as sc
import scvi

from benchmark_utils import add_cell_types_grouped, preprocess_scrna
from constants import N_GENES


# %% UMAP plot for train_dataset
use_full_adata = True # if False, use only the train adata (which is what the model was trained on)
# Load scRNAseq dataset
adata = sc.read("/home/owkin/project/cti/cti_adata.h5ad")
preprocess_scrna(adata,
                    keep_genes=N_GENES,
                    batch_key="donor_id")
cell_type = f"cell_types_grouped_2nd_level_granularity"
adata, train_test_index = add_cell_types_grouped(adata, "2nd_level_granularity")
train_test_index2 = pd.read_csv("/home/owkin/project/train_test_index_dataframes/train_test_index_matrix_common.csv", index_col=1).iloc[:,1:]
col_name = "primary_groups"
adata.obs[f"cell_types_grouped_1st_level_granularity"] = train_test_index2[col_name]
if use_full_adata:
    adata = adata[adata.obs["cell_types_grouped_1st_level_granularity"] != "To remove"]
    adata = adata[adata.obs["cell_types_grouped_2nd_level_granularity"] != "To remove"]
else:
    adata = adata[train_test_index["Train index"]]
filtered_genes = adata.var.index[adata.var["highly_variable"]].tolist()
adata = adata[:, filtered_genes]
# Load scvi model
scvi_model = scvi.model.SCVI.load("/home/owkin/project/Simon/plots_for_paper/scvi_model.pkl", adata=adata.copy())
adata.obsm["X_scVI"] = scvi_model.get_latent_representation()
# Compute UMAP on the scVI latent space
sc.pp.neighbors(adata, use_rep="X_scVI")
sc.tl.umap(adata)
# Plot the UMAP
sc.pl.umap(adata, color="cell_types_grouped_2nd_level_granularity")
sc.pl.umap(adata, color="cell_types_grouped_1st_level_granularity")  


# %% Plot the log of the metrics epoch after epoch
model_scvi_metrics = pd.read_csv("/home/owkin/project/Simon/plots_for_paper/mixupvi_ablation_training_metrics.csv")
model_mixupvi_metrics = pd.read_csv("/home/owkin/project/Simon/plots_for_paper/mixupvi_training_metrics.csv")
model_scvi_metrics["model"] = "scVI"
model_mixupvi_metrics["model"] = "MixUpVI"
all_metrics = pd.concat([model_scvi_metrics, model_mixupvi_metrics])
all_metrics["ELBO"] = all_metrics["kl_local_validation"] + all_metrics["reconstruction_loss_validation"]
all_metrics["pearson_coeff_deconv_validation"] = all_metrics["pearson_coeff_deconv_validation"].fillna(method="backfill")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
sns.lineplot(hue='model', x='epoch', y='ELBO', data=all_metrics, ax=axes[0])
axes[0].set_title('ELBO loss')
axes[0].set_ylabel('ELBO loss')
sns.lineplot(hue='model', x='epoch', y='pearson_coeff_validation', data=all_metrics, ax=axes[1])
axes[1].set_title('Pearson correlation between pseudobulks in the latent space')
axes[1].set_ylabel('Pearson correlation')
sns.lineplot(hue='model', x='epoch', y='pearson_coeff_deconv_validation', data=all_metrics, ax=axes[2])
axes[2].set_title('Pearson correlation between ground truth and estimated proportions in the latent space')
axes[2].set_ylabel('Pearson correlation')
plt.tight_layout()
plt.show()
# %%
