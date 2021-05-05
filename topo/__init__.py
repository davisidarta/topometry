from . import base
from . import diag
from . import spectral
from . import tpgraph as tpg
from . import layouts as lyt
from . import utils
from . import models as ml
from . import plot as pl
from ._utils import annotate_doc_types
from .version import __version__

import sys

sys.modules.update({f'{__name__}.{m}': globals()[m] for m in ['base', 'diag', 'spectral', 'tpg', 'lyt', 'utils', 'ml', 'plot']})

annotate_doc_types(sys.modules[__name__], 'topo')
del sys, annotate_doc_types