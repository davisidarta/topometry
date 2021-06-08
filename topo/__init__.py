import sys

from . import layouts as lt
from . import models as ml
from . import plot as pl
from . import spectral as spt
from . import tpgraph as tpg
from . import utils
from ._utils import annotate_doc_types
from .base import ann
from .version import __version__

sys.modules.update({f'{__name__}.{m}': globals()[m] for m in ['ann', 'spt', 'tpg', 'lt', 'utils', 'ml', 'pl']})

annotate_doc_types(sys.modules[__name__], 'topo')
del sys, annotate_doc_types