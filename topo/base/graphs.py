# Some utility functions for handling igraph objects

import pandas as pd
import numpy as np
import igraph as ig
from topo.tpgraph import cknn, diffusion, multiscale

def igraph_from_w_adj(A):
    G = ig.Graph.Weighted_Adjacency(A)
    return G

def igraph_from_adj(A):
    G = ig.Graph.Adjacency(A)
    return G

def graph_cl_components(G, mode='Strong'):
    cl = G.clusters(mode=mode)
    clusters = cl.membership
    return clusters

def graph_cl_louvain(G, weights):
    """
    Community structure based on the multilevel algorithm of Blondel et al.

    Parameters
    ----------
    weights
    	edge attribute name or a list containing edge weights

    Returns
    -------
        clusters
            clustering levels
    """
    weights = igraph_from_w_adj(weights)
    cl = G.community_multilevel(weights=weights)
    clusters = cl.membership
    return clusters

def graph_cl_walktrap(G, weights, steps=5):
    """
    Community detection algorithm of Latapy & Pons, based on random walks.

    Parameters
    ----------

    weights
    	name of an edge attribute or a list containing edge weights

    steps
    	length of random walks to perform

    Returns
    -------
        clusters
            clustering levels

    """
    weights = igraph_from_w_adj(weights)
    cl = G.community_walktrap(weights=weights, steps=steps)
    clusters = cl.membership
    return clusters

def graph_cl_leiden(G, weights,
    objective_function='CPM',
    resolution_parameter=0.5,
    beta=0.01,
    initial_membership=None,
    n_iterations=-1,
    node_weights=None):
    """
    Finds the community structure of the graph using the Leiden algorithm of Traag, van Eck & Waltman.

    Parameters
    ----------

    objective_function
    	whether to use the Constant Potts Model (CPM) or modularity. Must be either "CPM" or "modularity".
    weights
    	edge weights to be used. Can be a sequence or iterable or even an edge attribute name.
    resolution_parameter
    	the resolution parameter to use. Higher resolutions lead to more smaller communities, while lower resolutions lead to fewer larger communities.
    beta
    	parameter affecting the randomness in the Leiden algorithm. This affects only the refinement step of the algorithm.
    initial_membership
    	if provided, the Leiden algorithm will try to improve this provided membership. If no argument is provided, the aglorithm simply starts from the singleton partition.
    n_iterations
    	the number of iterations to iterate the Leiden algorithm. Each iteration may improve the partition further. Using a negative number of iterations will run until a stable iteration is encountered (i.e. the quality was not increased during that iteration).
    node_weights
    	the node weights used in the Leiden algorithm. If this is not provided, it will be automatically determined on the basis of whether you want to use CPM or modularity. If you do provide this, please make sure that you understand what you are doing.

    Returns
    ------------
        clusters
            clustering levels
    """



    weights = igraph_from_w_adj(weights)
    cl = G.community_leiden(weights=weights,
        objective_function=objective_function,
        resolution_parameter=resolution_parameter,
        beta=beta,
        initial_membership=initial_membership,
        n_iterations=n_iterations,
        node_weights=None
        )
    clusters = cl.membership
    return clusters

def global_transitivity(G, mode='nan'):
    """
    Calculates the global transitivity (clustering coefficient) of the graph.

    Parameters
    ---------------
    mode: str, 'nan' or 'zero'
        if TRANSITIVITY_ZERO or "zero", the result will be zero if the graph does not
        have any triplets. If "nan" or TRANSITIVITY_NAN, the result will be NaN
        (not a number).

    Returns
    --------------
    global transitivity

    """
    global_trans = G.transitivity_undirected
    return global_trans

def transitivity_local_undirected(G, vertices=None, mode='nan', weights=None):
    """
    Calculates the local transitivity (clustering coefficient) of the given vertices in the graph.

    The transitivity measures the probability that two neighbors of a vertex are connected. In case of the local transitivity, this probability is calculated separately for each vertex.

    Parameters
    --------------
    vertices
        a list containing the vertex IDs which should be included in the result. None means all of the vertices.
    mode
        defines how to treat vertices with degree less than two. If TRANSITIVITT_ZERO or "zero", these vertices will have zero transitivity. If TRANSITIVITY_NAN or "nan", these vertices will have NaN (not a number) as their transitivity.
    weights
    	edge weights to be used. Can be a sequence or iterable or even an edge attribute name.

    Returns
    ---------------
    the transitivities for the given vertices in a list


    """
    local_trans = G.transitivity_local_undirected(vertices=None, mode=mode, weights=weights)
