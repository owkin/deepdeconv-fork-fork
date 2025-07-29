"""Deconvolution benchmark methods classes.."""

from __future__ import annotations

from abc import abstractmethod
from loguru import logger
from TAPE import Deconvolution
from TAPE.deconvolution import ScadenDeconvolution
from pydeconv import SignatureMatrix
from pydeconv.model import OLS, NNLS, DWLS, RLR, NuSVR, WNNLS
import anndata as ad
import functools
import pandas as pd
import numpy as np
import scipy.sparse
import torch

from benchmark_utils.latent_signature_utils import create_latent_signature
from benchmark_utils.deconv_utils import log_scale_data, use_nnls_method
from benchmark_utils.pseudobulk_dataset_utils import create_anndata_pseudobulk
from benchmark_utils.training_utils import (
    fit_destvi,
    fit_scvi,
    fit_mixupvi,
)

class AbstractDeconvolutionMethod:
    """Abstract Deconvolution Method that every deconvolution method has to inherent from.

    Each deconvolution algorithm must contain an apply_deconvolution method. If the 
    algorithm needs fitting (e.g. MixUpVI), then it should be done in the __init__ method.
    """
    @abstractmethod 
    def apply_deconvolution(self, to_deconvolve: ad.AnnData, **kwargs):
        """Apply deconvolution method on data to deconvolve.
        
        Parameters
        ----------
        to_deconvolve: ad.AnnData
            The data to deconvolve.
        """
        
class MixUpVIMethod(AbstractDeconvolutionMethod):
    """MixUpVI deconvolution method."""
    def __init__(
        self, 
        adata_train: ad.AnnData, 
        cell_type_group: str, 
        model_path: str = "", 
        save_model: bool = False, 
    ):
        """Fit MixUpVI and create the latent signature matrix."""
        self.filtered_genes = adata_train.var.index[
            adata_train.var["highly_variable"]
        ].tolist()
        adata_train = adata_train[:,self.filtered_genes]
        self.adata_obs = adata_train.obs

        logger.debug("Fitting MixUpVI...")
        self.mixupvi = fit_mixupvi(
            adata=adata_train.copy(),
            model_path=model_path,
            cell_type_group=cell_type_group,
            save_model=save_model,
        )

        logger.debug("Training over. Creation of latent signature matrix...")
        self.adata_latent_signature = create_latent_signature(
            adata=adata_train,
            model=self.mixupvi,
            use_mixupvi=False, # should be equal to use_mixupvi, but if True, 
            # then it averages as many cells as self.n_cells_per-pseudobulk from mixupvae 
            # (and not the number we wish in the benchmark)
            average_all_cells = True,
        )
        self.adata_latent_signature = pd.DataFrame(
            self.adata_latent_signature.X.T,
            index=self.adata_latent_signature.var_names,
            columns=self.adata_latent_signature.obs["cell type"]
        )

    def apply_deconvolution(self, to_deconvolve: ad.AnnData|pd.DataFrame):
        """Apply the MixUpVI method on data to deconvolve."""
        if isinstance(to_deconvolve, ad.AnnData):
            # Pseudobulks constructed from scRNAseq
            obs_names = to_deconvolve.obs_names
            to_deconvolve = to_deconvolve[:, self.filtered_genes]
        elif isinstance(to_deconvolve, pd.DataFrame):
            # Bulk/FACS data
            obs_names = to_deconvolve.columns
            to_deconvolve = create_anndata_pseudobulk(
                adata_obs=self.adata_obs, 
                adata_var_names=self.filtered_genes,
                x=to_deconvolve.loc[self.filtered_genes].T.values
            )
        else:
            message = (
                "Data to deconvolve during inference can either be AnnData or DataFrame, "
                f"but here it is of type {type(to_deconvolve)}."
            )
            logger.error(message)
            raise ValueError(message)
        
        latent_adata = self.mixupvi.get_latent_representation(
            to_deconvolve, get_pseudobulk=False
        )
        latent_adata = pd.DataFrame(
            index=obs_names,
            columns=self.adata_latent_signature.index,
            data=latent_adata
        ).T
        deconvolution_results = use_nnls_method(
            latent_adata, self.adata_latent_signature
        )

        return deconvolution_results
    
class scVIMethod(AbstractDeconvolutionMethod):
    """scVI deconvolution method."""
    def __init__(
        self, 
        adata_train: ad.AnnData, 
        model_path: str = "", 
        save_model: bool = False, 
    ):
        """Fit scVI and create the latent signature matrix."""
        self.filtered_genes = adata_train.var.index[
            adata_train.var["highly_variable"]
        ].tolist()
        adata_train = adata_train[:,self.filtered_genes]
        self.adata_obs = adata_train.obs

        logger.debug("Fitting scVI...")
        self.scvi = fit_scvi(
            adata=adata_train.copy(),
            model_path=model_path,
            save_model=save_model,
        )

        logger.debug("Training over. Creation of latent signature matrix...")
        self.adata_latent_signature = create_latent_signature(
            adata=adata_train,
            model=self.scvi,
            use_mixupvi=False,
            average_all_cells = True,
        )
        self.adata_latent_signature = pd.DataFrame(
            self.adata_latent_signature.X.T,
            index=self.adata_latent_signature.var_names,
            columns=self.adata_latent_signature.obs["cell type"]
        )

    def apply_deconvolution(self, to_deconvolve: ad.AnnData|pd.DataFrame):
        """Apply the scVI method on data to deconvolve."""
        if isinstance(to_deconvolve, ad.AnnData):
            # Pseudobulks constructed from scRNAseq
            obs_names = to_deconvolve.obs_names
            to_deconvolve = to_deconvolve[:, self.filtered_genes]
        elif isinstance(to_deconvolve, pd.DataFrame):
            # Bulk/FACS data
            obs_names = to_deconvolve.columns
            to_deconvolve = create_anndata_pseudobulk(
                adata_obs=self.adata_obs, 
                adata_var_names=self.filtered_genes,
                x=to_deconvolve.loc[self.filtered_genes].T.values
            )
        else:
            message = (
                "Data to deconvolve during inference can either be AnnData or DataFrame, "
                f"but here it is of type {type(to_deconvolve)}."
            )
            logger.error(message)
            raise ValueError(message)
        
        latent_adata = self.scvi.get_latent_representation(to_deconvolve)
        latent_adata = pd.DataFrame(
            index=obs_names,
            columns=self.adata_latent_signature.index,
            data=latent_adata
        ).T
        deconvolution_results = use_nnls_method(
            latent_adata, self.adata_latent_signature
        )

        return deconvolution_results

class DestVIMethod(AbstractDeconvolutionMethod):
    """DestVI deconvolution method."""
    def __init__(
        self, 
        adata_train: ad.AnnData,
        adata_pseudobulk: ad.AnnData,
        model_path1: str = "",
        model_path2: str = "",
        cell_type_group: str = "cell_types_grouped",
        save_model: bool = False, 
    ):
        """Fit DestVI."""
        self.filtered_genes = adata_train.var.index[
            adata_train.var["highly_variable"]
        ].tolist()
        adata_train = adata_train[:,self.filtered_genes]

        logger.debug("Fitting DestVI...")
        _, self.destvi = fit_destvi(
            adata=adata_train.copy(),
            adata_pseudobulk=adata_pseudobulk[:,self.filtered_genes],
            model_path_1=model_path1, 
            model_path_2=model_path2, 
            cell_type_group=cell_type_group,
            save_model=save_model, 
        )

        logger.debug("Training over.")

    def apply_deconvolution(self, to_deconvolve: ad.AnnData|pd.DataFrame):
        """Apply the DestVI method on data to deconvolve."""
        if isinstance(to_deconvolve, ad.AnnData):
            # Pseudobulks constructed from scRNAseq
            to_deconvolve = to_deconvolve[:, self.filtered_genes]
        elif isinstance(to_deconvolve, pd.DataFrame):
            # Bulk/FACS data
            to_deconvolve = create_anndata_pseudobulk(
                adata_obs=self.adata_obs, 
                adata_var_names=self.filtered_genes,
                x=to_deconvolve.loc[self.filtered_genes].T.values
            )
        else:
            message = (
                "Data to deconvolve during inference can either be AnnData or DataFrame, "
                f"but here it is of type {type(to_deconvolve)}."
            )
            logger.error(message)
            raise ValueError(message)

        deconvolution_results = self.destvi.get_proportions(to_deconvolve)
        deconvolution_results = deconvolution_results.drop(["noise_term"], axis=1)

        return deconvolution_results
    

def _patch_torch_load():
    """Patch torch.load to use weights_only=False by default for TAPE compatibility"""
    original_load = torch.load
    @functools.wraps(original_load)
    def patched_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    torch.load = patched_load

_patch_torch_load()

class TAPEMethod(AbstractDeconvolutionMethod):
    """TAPE deconvolution method."""
    def __init__(
        self,
        signature_matrix_name: str,
        signature_matrix: pd.DataFrame,
        cell_type_group: str,
        adata_train: ad.AnnData = None,
        use_signature: bool = True,
    ):
        self.signature_matrix_name = signature_matrix_name
        self.use_signature = use_signature

        if use_signature or adata_train is None:
            self.training_dataset = signature_matrix
        else:
            # Gene filtering step
            filtered_genes = adata_train.var.index[adata_train.var["highly_variable"]].tolist()
            adata_train = adata_train[:, filtered_genes]
            if cell_type_group not in adata_train.obs.columns:
                raise ValueError(f"Cell type column '{cell_type_group}' not found in adata_train.obs")
            self.training_dataset = pd.DataFrame(
                adata_train.layers["counts"].toarray(),
                index=adata_train.obs[cell_type_group].values,
                columns=adata_train.var_names
            )

    def apply_deconvolution(self, to_deconvolve: ad.AnnData | pd.DataFrame):
        """Apply the TAPE method on data to deconvolve."""
        if isinstance(to_deconvolve, ad.AnnData):
            to_deconvolve = pd.DataFrame(
                index=to_deconvolve.obs_names,
                columns=to_deconvolve.var_names,
                data=to_deconvolve.layers["counts"],
            ).T
        elif not isinstance(to_deconvolve, pd.DataFrame):
            message = (
                "Data to deconvolve during inference can either be AnnData or DataFrame, "
                f"but here it is of type {type(to_deconvolve)}."
            )
            logger.error(message)
            raise ValueError(message)

        _, deconvolution_results = Deconvolution(
            self.training_dataset,
            to_deconvolve.T,
            sep="\t",
            scaler="mms",
            datatype="counts",
            genelenfile=None,
            mode="overall",
            adaptive=True,
            variance_threshold=0.98,
            save_model_name=None,
            batch_size=128,
            epochs=128,
            seed=1,
        )

        return deconvolution_results

class ScadenMethod(AbstractDeconvolutionMethod):
    """Apply the Scaden method on data to deconvolve."""
    def __init__(
        self,
        signature_matrix_name: str,
        signature_matrix: pd.DataFrame,
        cell_type_group: str,
        adata_train: ad.AnnData = None,
        use_signature: bool = True,
    ):
        self.signature_matrix_name = signature_matrix_name
        self.use_signature = use_signature

        if use_signature or adata_train is None:
            self.training_dataset = signature_matrix
        else:
            # Gene filtering step
            filtered_genes = adata_train.var.index[adata_train.var["highly_variable"]].tolist()
            adata_train = adata_train[:, filtered_genes]
            if cell_type_group not in adata_train.obs.columns:
                raise ValueError(f"Cell type column '{cell_type_group}' not found in adata_train.obs")
            self.training_dataset = pd.DataFrame(
                adata_train.layers["counts"].toarray(),
                index=adata_train.obs[cell_type_group].values,
                columns=adata_train.var_names
            )

    def apply_deconvolution(self, to_deconvolve: ad.AnnData | pd.DataFrame):
        """Apply the Scaden method on data to deconvolve."""
        if isinstance(to_deconvolve, ad.AnnData):
            to_deconvolve = pd.DataFrame(
                index=to_deconvolve.obs_names,
                columns=to_deconvolve.var_names,
                data=to_deconvolve.layers["counts"],
            ).T
        elif not isinstance(to_deconvolve, pd.DataFrame):
            message = (
                "Data to deconvolve during inference can either be AnnData or DataFrame, "
                f"but here it is of type {type(to_deconvolve)}."
            )
            logger.error(message)
            raise ValueError(message)

        deconvolution_results = ScadenDeconvolution(
            self.training_dataset,
            to_deconvolve.T,
            sep="\t",
            batch_size=128,
            epochs=128,
        )

        return deconvolution_results
    

################ All Pydeconv methods ################

class PydeconvBaseMethod(AbstractDeconvolutionMethod):
    """Base class for pydeconv-based methods."""
    def __init__(self, signature_matrix_name: str, signature_matrix: pd.DataFrame, use_log_scale: bool = False):
        self.signature_matrix_name = signature_matrix_name
        self.use_log_scale = use_log_scale
        
        # Apply log scaling if requested
        if self.use_log_scale:
            logger.debug(f"Applying log scaling to signature matrix for {signature_matrix_name}")
            try:
                signature_matrix = log_scale_data(signature_matrix)
            except Exception as e:
                logger.error(f"Error applying log scaling to signature matrix: {str(e)}")
                raise
            
        # Convert pandas DataFrame to SignatureMatrix and ensure numpy array
        try:
            self.signature_matrix = SignatureMatrix(signature_matrix.astype(np.float64))
        except Exception as e:
            logger.error(f"Error converting signature matrix to SignatureMatrix: {str(e)}")
            raise
            
        self.solver = None  # Will be set by child classes
        
    def apply_deconvolution(self, to_deconvolve: ad.AnnData|pd.DataFrame):
        if isinstance(to_deconvolve, ad.AnnData):
            # Ensure data is in correct format
            if isinstance(to_deconvolve.X, np.ndarray) or scipy.sparse.issparse(to_deconvolve.X):
                data = to_deconvolve.X
                if self.use_log_scale:
                    logger.debug("Applying log scaling to AnnData.X")
                    data = log_scale_data(data)
                to_deconvolve.X = data
                
            if "counts" in to_deconvolve.layers:
                data = to_deconvolve.layers["counts"]
                if self.use_log_scale:
                    logger.debug("Applying log scaling to AnnData.layers['counts']")
                    data = log_scale_data(data)
                to_deconvolve.layers["counts"] = data
        else:
            # For DataFrame input, convert to AnnData
            data = to_deconvolve.T
            if self.use_log_scale:
                logger.debug("Applying log scaling to DataFrame input")
                data = log_scale_data(data)
            adata = ad.AnnData(data)
            adata.layers["counts"] = adata.X
            to_deconvolve = adata
            
        cell_prop = self.solver.fit_transform(to_deconvolve, layer="counts", ratio=True)
        return pd.DataFrame(cell_prop, index=to_deconvolve.obs_names, columns=self.signature_matrix.list_cell_types)


class OLSMethod(PydeconvBaseMethod):
    """OLS method using pydeconv implementation."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.solver = OLS(self.signature_matrix)

class DWLSMethod(PydeconvBaseMethod):
    """DWLS method using pydeconv implementation."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.solver = DWLS(self.signature_matrix)

class NNLSMethod(PydeconvBaseMethod):
    """NNLS method using pydeconv implementation."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.solver = NNLS(self.signature_matrix)

class RLRMethod(PydeconvBaseMethod):
    """Ridge Linear Regression method using pydeconv implementation."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.solver = RLR(self.signature_matrix)

class NuSVRMethod(PydeconvBaseMethod):
    """Nu-Support Vector Regression method using pydeconv implementation."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.solver = NuSVR(self.signature_matrix)

class WNNLSMethod(PydeconvBaseMethod):
    """Weighted Non-Negative Least Squares method using pydeconv implementation."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.solver = WNNLS(self.signature_matrix)

class RLRMethod(PydeconvBaseMethod):
    """Ridge Linear Regression method using pydeconv implementation."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.solver = RLR(self.signature_matrix)

class NuSVRMethod(PydeconvBaseMethod):
    """Nu-Support Vector Regression method using pydeconv implementation."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.solver = NuSVR(self.signature_matrix)

class WNNLSMethod(PydeconvBaseMethod):
    """Weighted Non-Negative Least Squares method using pydeconv implementation."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.solver = WNNLS(self.signature_matrix)