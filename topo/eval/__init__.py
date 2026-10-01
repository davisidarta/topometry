from .global_scores import global_score_pca, global_score_laplacian
from .local_scores import knn_spearman_r, knn_kendall_tau, geodesic_distance, geodesic_correlation
# Re-exported for backwards compatibility: TopOMetry's own implementation was removed
# in favour of scikit-learn's in 4974acf5 (2023-07-05), but `topo.eval.trustworthiness`
# had been documented and used, so the name is kept available here.
from sklearn.manifold import trustworthiness
from .rmetric import RiemannMetric, get_eccentricity
