import sys
from .base import ann
from . import layouts as lt
from .topograph import TopOGraph
from . import plot as pl
from . import spectral as spt
from . import tpgraph as tpg
from . import eval
from . import utils
from . import pipes
from .utils._utils import read_pkl, annotate_doc_types

from .version import __version__

sys.modules.update({f'{__name__}.{m}': globals()[m] for m in ['ann', 'lt', 'TopOGraph', 'pl', 'spt', 'tpg', 'eval',
                                                              'pipes', 'read_pkl']})

annotate_doc_types(sys.modules[__name__], 'topo')
del sys, annotate_doc_types