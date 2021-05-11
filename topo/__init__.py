from . import base
from . import diag as dx
from . import spectral as spt
from . import tpgraph as tpg
from . import layouts as lt
from . import utils
from . import models as ml
from . import plot as pl

from ._utils import annotate_doc_types
from .version import __version__

import sys

sys.modules.update({f'{__name__}.{m}': globals()[m] for m in ['base', 'dx', 'spt', 'tpg', 'lt', 'utils', 'ml', 'pl']})

annotate_doc_types(sys.modules[__name__], 'topo')
del sys, annotate_doc_types