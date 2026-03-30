# ModalityBridge: kNN-based cross-modality imputation for unpaired multi-omic data
# Author: Generated for topometry multi-omic integration

import copy
import numpy as np


class ModalityBridge:
    """
    Stores a fitted modality bridge for imputing RNA expression from ATAC LSI.

    Serializable via joblib. The bridge contains:
      - ATAC LSI coordinates of paired reference cells (for neighbor lookup)
      - HVG-subsetted RNA expression of paired reference cells (for imputation)
      - hnswlib index (for fast nearest neighbor search)
      - Metadata (k, metric, HVG names, lsi parameters)

    The bridge does NOT store any linear transformation, regression weights,
    or dimensionality-reduced RNA representation.
    """

    def __init__(self):
        self.lsi_coords = None       # ndarray n_paired x n_lsi_components
        self.rna_expression = None   # ndarray n_paired x n_hvgs (raw normalized, not PCs)
        self.hvg_names = None        # list of HVG names
        self.k = None
        self.metric = None
        self.lsi_params = None       # dict from adata_atac.uns["lsi"]
        self._hnsw_index = None      # hnswlib index, rebuilt on load
        self.is_fitted = False

    def save(self, path: str):
        """Serialize to disk using joblib."""
        import joblib
        bridge_copy = copy.deepcopy(self)
        bridge_copy._hnsw_index = None
        joblib.dump(bridge_copy, path)

    @classmethod
    def load(cls, path: str) -> "ModalityBridge":
        import joblib
        bridge = joblib.load(path)
        bridge._rebuild_index()
        return bridge

    def _rebuild_index(self):
        """Rebuild hnswlib index from stored lsi_coords."""
        if self.lsi_coords is None:
            return
        try:
            import hnswlib
        except ImportError:
            raise ImportError(
                "ModalityBridge requires hnswlib for nearest-neighbor search. "
                "Install with: pip install hnswlib"
            )
        dim = self.lsi_coords.shape[1]
        n = self.lsi_coords.shape[0]
        metric_map = {'euclidean': 'l2', 'cosine': 'cosine'}
        space = metric_map.get(self.metric, 'l2')

        index = hnswlib.Index(space=space, dim=dim)
        index.init_index(max_elements=n, ef_construction=200, M=30)
        index.add_items(self.lsi_coords.astype(np.float32))
        index.set_ef(200)
        self._hnsw_index = index
