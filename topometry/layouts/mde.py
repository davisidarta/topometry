# Functions for using Minimum Distortion Embedding(MDE) for graph layout.
# The MDE algorithm was brilliantly coinceived by Akshay Agrawal in
# the monograph https://arxiv.org/abs/2103.02559

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import pymde

