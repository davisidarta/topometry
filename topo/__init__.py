import sys
try:
    import scanpy
    _HAVE_SCANPY = True
except ImportError:
    _HAVE_SCANPY = False
from .base import ann
from . import layouts as lt
from .topograph import TopOGraph, load_topograph, save_topograph
from . import plot as pl
from . import spectral as spt
from . import tpgraph as tpg
from . import eval
from . import utils
from . import pipes
if _HAVE_SCANPY:
    from . import single_cell as sc

from .version import __version__
if _HAVE_SCANPY:
    sys.modules.update({f'{__name__}.{m}': globals()[m] for m in ['ann', 'lt', 'TopOGraph', 'pl', 'spt', 'tpg', 'eval',
                                                              'pipes', 'load_topograph', 'save_topograph', 'sc']})
else:
    sys.modules.update({f'{__name__}.{m}': globals()[m] for m in ['ann', 'lt', 'TopOGraph', 'pl', 'spt', 'tpg', 'eval',
                                                              'pipes', 'load_topograph', 'save_topograph']})
del sys