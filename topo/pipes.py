import matplotlib
matplotlib.use('Agg')  # plotting backend compatible with screen
import sys
from topo.models import TopOGraph

filename = sys.argv[1]  # read filename from command line

def TopoMAP(data, *tg_kwargs, **map_kwargs):
    """""""""
    Easy, direct application of topological graphs (``TopoGraph``) and Manifold Approximation and Projection (``MAP``) for layout optimization with triple
    approximation of the Laplace-Beltrami Operator.

    Parameters
    ----------
    data: np.ndarray, csr_matrix, pd.DataFrame
        Input data. Will be converted to csr_matrix by default

    *tg_kwargs: dict
        keyword arguments for the ``TopoGraph`` instance.

    **map_kwargs: dict
        keyword arguments for the ``MAP`` function

    Returns
    -------
    TopoGraph:
        object containing topological graph analysis
    embedding: np.ndarray
        lower dimensional projection for optimal visualization

    """""""""

    tg = TopOGraph(*tg_kwargs).fit(data)
    graph = tg.transform()
    basis = tg.MSDiffMap
    emb = tg.MAP(basis, graph, **map_kwargs)

    return tg, emb

def TopoMDE(data, *tg_kwargs, **mde_kwargs):
    """""""""
    Easy, direct application of topological graphs (``TopoGraph``) and Manifold Approximation and Projection (``MAP``) for layout optimization with triple
    approximation of the Laplace-Beltrami Operator.

    Parameters
    ----------
    data: np.ndarray, csr_matrix, pd.DataFrame
        Input data. Will be converted to csr_matrix by default

    *tg_kwargs: dict
        keyword arguments for the ``TopoGraph`` instance.

    **mde_kwargs: dict
        keyword arguments for the ``MDE`` function

    Returns
    -------
    TopoGraph:
        object containing topological graph analysis
    embedding: np.ndarray
        lower dimensional projection for optimal visualization

    """""""""

    tg = TopOGraph(*tg_kwargs).fit(data)
    graph = tg.transform()
    topo = tg.MSDiffMap

    emb = tg.MDE(graph, **mde_kwargs)

    return tg, emb
