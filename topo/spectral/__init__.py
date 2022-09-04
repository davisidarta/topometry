from ._spectral import graph_laplacian, diffusion_operator, DM, LE, degree
from .umap_layouts import optimize_layout_euclidean, optimize_layout_generic, optimize_layout_inverse, optimize_layout_aligned_euclidean
from .ies import find_independent_coordinates, calc_m2_score, compute_tangent_plane
from .eigen import EigenDecomposition, eigendecompose