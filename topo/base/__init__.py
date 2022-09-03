from .ann import kNN

try:
    import numba
    _have_numba = True
except ImportError:
    _have_numba = False

if _have_numba:
    from .dists import *
    from .sparse import *
