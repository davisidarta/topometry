import inspect
from functools import partial
from types import ModuleType, MethodType
from typing import Union, Callable, Optional
from weakref import WeakSet



def read_pkl(wd=None, filename='topograph.pkl'):
    try:
        import pickle
    except ImportError:
        return (print('Pickle is needed for loading the TopOGraph. Please install it with `pip3 install pickle`'))

    if wd is None:
        import os
        wd = os.getcwd()
    with open(wd + filename, 'rb') as input:
        TopOGraph = pickle.load(input)
    return TopOGraph

def _one_of_ours(obj, root: str):
    return (
        hasattr(obj, "__name__")
        and not obj.__name__.split(".")[-1].startswith("_")
        and getattr(
            obj, '__module__', getattr(obj, '__qualname__', obj.__name__)
        ).startswith(root)
    )

def descend_classes_and_funcs(mod: ModuleType, root: str, encountered=None):
    if encountered is None:
        encountered = WeakSet()
    for obj in vars(mod).values():
        if not _one_of_ours(obj, root):
            continue
        if callable(obj) and not isinstance(obj, MethodType):
            yield obj
            if isinstance(obj, type):
                for m in vars(obj).values():
                    if callable(m) and _one_of_ours(m, root):
                        yield m
        elif isinstance(obj, ModuleType) and obj not in encountered:
            if obj.__name__.startswith('scanpy.tests'):
                # Python’s import mechanism seems to add this to `scanpy`’s attributes
                continue
            encountered.add(obj)
            yield from descend_classes_and_funcs(obj, root, encountered)

def getdoc(c_or_f: Union[Callable, type]) -> Optional[str]:
    if getattr(c_or_f, '__doc__', None) is None:
        return None
    doc = inspect.getdoc(c_or_f)
    if isinstance(c_or_f, type) and hasattr(c_or_f, '__init__'):
        sig = inspect.signature(c_or_f.__init__)
    else:
        sig = inspect.signature(c_or_f)

    def type_doc(name: str):
        param: inspect.Parameter = sig.parameters[name]
        cls = getattr(param.annotation, '__qualname__', repr(param.annotation))
        if param.default is not param.empty:
            return f'{cls}, optional (default: {param.default!r})'
        else:
            return cls

    return '\n'.join(
        f'{line} : {type_doc(line)}' if line.strip() in sig.parameters else line
        for line in doc.split('\n')
    )

def annotate_doc_types(mod: ModuleType, root: str):
    for c_or_f in descend_classes_and_funcs(mod, root):
        c_or_f.getdoc = partial(getdoc, c_or_f)

# Some other utility functions
import numpy as np
from scipy.sparse import coo_matrix

def get_sparse_matrix_from_indices_distances(knn_indices, knn_dists, n_obs, n_neighbors):
    rows = np.zeros((n_obs * n_neighbors), dtype=int)
    cols = np.zeros((n_obs * n_neighbors), dtype=int)
    vals = np.zeros((n_obs * n_neighbors), dtype=float)
    for i in range(knn_indices.shape[0]):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            if knn_indices[i, j] == i:
                val = 0.0
            else:
                val = knn_dists[i, j]

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val

    result = coo_matrix((vals, (rows, cols)),
                        shape=(n_obs, n_obs))
    result.eliminate_zeros()
    return result.tocsr()

def get_indices_distances_from_sparse_matrix(X, n_neighbors):
    """
    Get the knn indices and distances for each point in a sparse k-nearest-neighbors matrix.

    Parameters
    ----------
    X : sparse matrix
        Input knn matrix to get indices and distances from.
    
    n_neighbors : int
        Number of neighbors to get.
    
    Returns
    -------
    knn_indices : ndarray of shape (n_obs, n_neighbors)
        The indices of the nearest neighbors for each point.
    
    knn_dists : ndarray of shape (n_obs, n_neighbors)
        The distances to the nearest neighbors for each point.
    """
    _knn_indices = np.zeros((X.shape[0], n_neighbors), dtype=int)
    _knn_dists = np.zeros(_knn_indices.shape, dtype=float)
    for row_id in range(X.shape[0]):
        # Find KNNs row-by-row
        row_data = X[row_id].data
        row_indices = X[row_id].indices
        if len(row_data) < n_neighbors: 
            raise ValueError(
                "Some rows contain fewer than n_neighbors distances!"
            )
        row_nn_data_indices = np.argsort(row_data)[: n_neighbors]
        _knn_indices[row_id] = row_indices[row_nn_data_indices]
        _knn_dists[row_id] = row_data[row_nn_data_indices]
    return _knn_indices, _knn_dists
