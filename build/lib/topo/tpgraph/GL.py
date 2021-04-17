# NOTICE: This draws from work from
# https://github.com/TheGravLab/A-Unifying-Framework-for-Spectrum-Preserving-Graph-Sparsification-and-Coarsening
# IMPLEMENTING A ROBUST ALGORITHM FOR GRAPH COARSENING ANS SPARSIFICATION
# FOR MORE INFORMATION, SEE
# https://papers.nips.cc/paper/2019/file/cd474f6341aeffd65f93084d0dae3453-Paper.pdf
# THE ORIGINAL IMPLEMENTATION DOES NOT SPECIFY A LICENSE, AND THUS
# THIS VERSION IS BOUND BY THE STANDARD MIT LICENSE OF PHENOMETRY


import numpy as np
import random
import time
from igraph import *

diagnosticSwitch = 0  # 0: prints nothing, 1: prints what its doing, 2: prints timing information

flatten = lambda l: [item for sublist in l for item in sublist]


def InvLaplGraph(edges,
                 reductionTarget='edges',
                 actionSwitch='both',
                 numSamplesS='all',
                 qOverS=1.0 / 8,
                 minProbPerActionD=1.0 / 4,
                 minTargetItems=1024,
                 plotError=False,
                 reproject=False):
    """

    Parameters
    ----------
    edges: X
        Target basis to reduce.
    reductionTarget: str, 'edges' or 'nodes', default 'edges'
        Target item to reduce.
    actionSwitch: str, 'both' or 'delete', default 'both'
        Choosing 'delete' does not allow contraction.
    numSamplesS: int or 'all', default 'all'
        Perturbed edges per sampled edges. Setting to 0 gives q=1 per round using the single-edge method.
    qOverS: float, from 0.0 to 1.0, default 1.0 / 8
        Perturbed edges per sampled edges. Setting to 0 gives q=1 per round using the single-edge method.
    minProbPerActionD: float, from 0.0 to 1.0, default 1.0 / 4
        Minimum expected (target items removed)/(num actions taken).
    minTargetItems: int or 'all', default 1024
        Minimum expected (target items removed)/(num actions taken).
    plotError: bool, default False
        Decide whether or not to compute the hyperbolic alignment of the output of the original eigenvectors.
    project: bool, default False
        Whether to project the reduced node-weighted laplacian appropriately projected back to
        # $V \times V$. Use this to get approximate solutions to your Lx=b problems.

    Returns
    -------
    GLGraph containing the following:
        GLGraph.reducedLaplacian
            Reduced node-weighted laplacian of size $\tilde{V} \times \tilde{V}$
        GLGraph.reducedLaplacianOriginalDimension
            The reduced node-weighted laplacian appropriately projected back to the original dimensions.
            Use this to get approximate solutions to your Lx=b problems.

    """

    print("Starting")
    flatten = lambda l: [item for sublist in l for item in
                         sublist]  # Because I couldn't find a general 'flatten' in Python.
    if minTargetItems == 'none':
        if reductionTarget == 'nodes':
            minTargetItems = 2
        elif reductionTarget == 'edges' and actionSwitch == 'both':
            minTargetItems = 1
        elif reductionTarget == 'edges' and actionSwitch == 'delete':
            minTargetItems = len(set(flatten(edges)))
    connected = False  # In case the graph becomes disconnected, try again.
    while not connected:  # In case the graph becomes disconnected, try again.
        print("Initializing GLGraph")
        g = GLGraph(edges, edgeWeights='none', nodeWeights='none', plotError=plotError, layout='random')
        edgeNumList = []  # List of edges in the reduced graphs if target is 'edges'.
        nodeNumList = []  # List of nodes in the reduced graphs if target is 'nodes'.
        eigenAlignList = []  # List of hyperbolic distance of eigenvector output if plotError is True.
        edgeNumList.append(len(g.edgeList))
        nodeNumList.append(len(g.nodeList))
        if plotError: eigenAlignList.append(g.get_eigenvector_alignment()[1:])
        iteration = 0
        while True:
            iteration += 1
            if np.mod(iteration, 1) == 0:  # Say where we are in the reduction.
                print("Iteration ", iteration, ", ", len(g.edgeList), "/", len(g.edgeListIn), " edges, ",
                      len(g.nodeList),
                      "/", len(g.nodeListIn), " nodes")
            if qOverS > 0:  # If q is determined by a fraction of s, use reduce_graph_multi_edge, as the edges should form a matching.
                g.reduce_graph_multi_edge(numSamples=numSamplesS, qFraction=qOverS, pMin=minProbPerActionD, \
                                          reductionType=actionSwitch, reductionTarget=reductionTarget,
                                          maxReweightFactor=0)
            else:  # If q is fixed at 1 (ie, qOverS==0), use reduce_graph_single_edge, as we do not care if the edges form a matching.
                g.reduce_graph_single_edge(minSamples=numSamplesS, pMin=minProbPerActionD, \
                                           reductionType=actionSwitch, reductionTarget=reductionTarget,
                                           maxReweightFactor=0)
            if reductionTarget == 'nodes':  # If targeting nodes, save data whenever the number of nodes is reduced.
                if len(g.nodeList) < nodeNumList[-1]:
                    edgeNumList.append(len(g.edgeList))
                    nodeNumList.append(len(g.nodeList))
                    if plotError: eigenAlignList.append(g.get_eigenvector_alignment()[1:])
            if reductionTarget == 'edges':  # If targeting edges, save data whenever the number of edges is reduced.
                if len(g.edgeList) < edgeNumList[-1]:
                    edgeNumList.append(len(g.edgeList))
                    nodeNumList.append(len(g.nodeList))
                    if plotError: eigenAlignList.append(g.get_eigenvector_alignment()[1:])
            if actionSwitch == 'both':  # If we can merge nodes, go until there are only two left.
                if len(g.nodeList) < 3 or (reductionTarget == 'edges' and len(g.edgeList) < minTargetItems) \
                        or (reductionTarget == 'nodes' and len(g.nodeList) < minTargetItems): break
            if actionSwitch == 'delete':  # If we cannot merge nodes, go until we have a spanning tree.
                if len(g.edgeList) < len(g.nodeList) or (
                        reductionTarget == 'edges' and len(g.edgeList) < minTargetItems): break
        iGraph = Graph()
        iGraph.add_vertices(g.nodeList)
        iGraph.add_edges(g.edgeList)
        if not iGraph.is_connected():
            print("Whoops! Disconnected. Retrying.")
        else:
            connected = True
        g.update_inverse_laplacian()
        if reproject:
            g.reducedLaplacianProjected = g.project_reduced_to_original(g.nodeWeightedInverseLaplacian)
        return g

class GLGraph(object):
    def __init__(self, edges, edgeWeights='none', nodeWeights='none', plotError=False, layout='random'):
        if diagnosticSwitch > 0: print('Making GLGraph')
        startTime = time.time()
        self.thereIsAProblem = False
        self.edgeListIn = np.array(edges)
        self.nodeListIn = sorted(list(set(flatten(self.edgeListIn))))

        if len(np.shape(edgeWeights)) == 0:
            self.edgeWeightsIn = np.ones(len(self.edgeListIn))
        else:
            self.edgeWeightsIn = np.array(edgeWeights)

        if len(np.shape(nodeWeights)) == 0:
            self.nodeWeightsIn = np.ones(len(self.nodeListIn))
        else:
            self.nodeWeightsIn = np.array(nodeWeights)

        # setting up the reduced graph
        self.edgeList = np.copy(self.edgeListIn)
        self.nodeList = np.copy(self.nodeListIn)
        self.edgeWeightList = np.copy(self.edgeWeightsIn)
        self.nodeWeightList = np.copy(self.nodeWeightsIn)
        self.nodeWeightListOld = np.copy(self.nodeWeightsIn)

        # making matrices
        if diagnosticSwitch > 0: print('making matrices')
        self.adjacency = self.make_adjacency(self.edgeListIn, self.nodeListIn, self.edgeWeightsIn)
        self.laplacian = self.adjacency_to_laplacian(self.adjacency)
        self.nodeWeightedLaplacianIn = (((self.laplacian).T) / self.nodeWeightsIn).T
        self.nodeWeightedLaplacian = np.copy(self.nodeWeightedLaplacianIn)
        self.jMatIn = np.outer(np.ones(len(self.nodeWeightsIn)), self.nodeWeightsIn) / np.sum(self.nodeWeightsIn)
        self.jMat = np.copy(self.jMatIn)
        self.contractedNodesToNodes = np.identity(len(self.nodeListIn))

        # initializing layout
        if diagnosticSwitch > 0: print('making layout')
        if len(np.shape(layout)) == 0:
            if layout == 'random':
                self.layout = np.array([tuple(np.random.random(2)) for item in range(len(self.nodeListIn))])
                self.boundaries = np.array([[0.0, 0.0], [1.0, 1.0]])
            else:
                # making igraph object
                if diagnosticSwitch > 0: print('making graph')
                import igraph as ig
                self.igraphIn = ig.Graph()
                (self.igraphIn).add_vertices(self.nodeListIn)
                (self.igraphIn).add_edges(self.edgeListIn)
                self.layout = (self.igraphIn).layout(layout)
                self.boundaries = np.array((self.layout).boundaries())
                boundaryTemp = np.max([np.max(self.boundaries), np.max(-self.boundaries)])
                self.boundaries = np.array([[-boundaryTemp, -boundaryTemp], [boundaryTemp, boundaryTemp]])
        else:
            boundaryTemp = np.max([np.max(layout), -np.min(layout)])
            self.boundaries = np.array([[-boundaryTemp, -boundaryTemp], [boundaryTemp, boundaryTemp]])
            self.layout = np.array([tuple(item) for item in layout])

        # computing the inverse and initial eigenvectors
        if diagnosticSwitch > 0: print('making inverses')
        self.nodeWeightedInverseLaplacianIn = self.invert_laplacian(self.nodeWeightedLaplacianIn, self.jMat)
        self.nodeWeightedInverseLaplacian = np.copy(self.nodeWeightedInverseLaplacianIn)
        if not plotError:
            self.eigenvaluesIn = np.zeros(len(self.nodeWeightedLaplacianIn))
            self.eigenvectorsIn = np.zeros(np.shape(self.nodeWeightedLaplacianIn))
        else:
            eigenvaluesTemp, eigenvectorsTemp = np.linalg.eig(self.nodeWeightedLaplacianIn)
            orderTemp = np.argsort(eigenvaluesTemp)
            self.eigenvaluesIn = eigenvaluesTemp[orderTemp]
            self.eigenvectorsIn = eigenvectorsTemp.T[orderTemp]
        self.originalEigenvectorOutput = np.array(
            [np.dot(self.nodeWeightedInverseLaplacianIn, eigVec) for eigVec in self.eigenvectorsIn])
        self.updatedInverses = True
        self.updateList = []
        self.rowsToDelete = []

        endTime = time.time()
        if diagnosticSwitch > 1: print('__init__: ', endTime - startTime)

    def make_adjacency(self, edgeListIn, nodeListIn=['none'], edgeWeightListIn=['none']):
        if np.any([item == 'none' for item in nodeListIn]):
            nodeListTemp = sorted(list(set(flatten(edgeListIn))))
        else:
            nodeListTemp = np.array(nodeListIn)
        if np.any([item == 'none' for item in edgeWeightListIn]):
            edgeWeightListTemp = np.ones(len(edgeListIn))
        else:
            edgeWeightListTemp = np.array(edgeWeightListIn)

        adjOut = np.zeros((len(nodeListTemp), len(nodeListTemp)))
        for index, edge in enumerate(edgeListIn):
            position0 = list(nodeListTemp).index(edge[0])
            position1 = list(nodeListTemp).index(edge[1])
            adjOut[position0, position1] += edgeWeightListTemp[index]
            adjOut[position1, position0] += edgeWeightListTemp[index]
        return adjOut

    def adjacency_to_laplacian(self, adjIn):
        lapOut = np.copy(-adjIn)
        for index in range(len(adjIn)):
            lapOut[index, index] = -np.sum(lapOut[index])
        return lapOut

    def invert_laplacian(self, lapIn, jMatIn):
        return np.linalg.inv(lapIn + jMatIn) - jMatIn

    def hyperbolic_distance(self, vector0, vector1):
        hyperbolicDistance = np.arccosh(1.0 + (np.linalg.norm(np.array(vector1) - np.array(vector0))) ** 2 / (
                    2 * np.dot(np.array(vector0), np.array(vector1))))
        if np.isnan(hyperbolicDistance):
            print("NAN in compare_vectors")
        return hyperbolicDistance

    def project_reduced_to_original(self, matIn):
        return np.dot(self.contractedNodesToNodes,
                      np.dot(matIn, np.dot(np.diag(1.0 / self.nodeWeightList), self.contractedNodesToNodes.T)))

    def get_eigenvector_alignment(self):
        if not self.updatedInverses:
            self.update_inverse_laplacian()
        distanceListOut = np.zeros(len(self.originalEigenvectorOutput))
        projectedNodeWeightedInverseLaplacian = self.project_reduced_to_original(self.nodeWeightedInverseLaplacian)
        for index in range(len(self.originalEigenvectorOutput)):
            distanceListOut[index] = self.hyperbolic_distance(self.originalEigenvectorOutput[index],
                                                              np.dot(projectedNodeWeightedInverseLaplacian,
                                                                     self.eigenvectorsIn[index]))
        return distanceListOut

    def make_wOmega_m_tau(self, method='random', numSamples='all'):
        startTime = time.time()
        if method == 'random':
            if numSamples == 'all':
                edgesToSample = range(len(self.edgeWeightList))
            elif numSamples >= len(self.edgeWeightList):
                edgesToSample = range(len(self.edgeWeightList))
            else:
                edgesToSample = sorted(np.random.choice(len(self.edgeWeightList), numSamples, replace=False))
        elif method == 'RM':
            edgesToSample = self.get_edgeList_proposal_RM(numSamples)
        effectiveResistanceOut = np.zeros(len(edgesToSample))
        edgeImportanceOut = np.zeros(len(edgesToSample))
        numTrianglesOut = np.zeros(len(edgesToSample))

        for index, edgeNum in enumerate(edgesToSample):
            vertex0 = self.edgeList[edgeNum][0]
            vertex1 = self.edgeList[edgeNum][1]
            invDotUTemp = self.nodeWeightedInverseLaplacian[:, vertex0] / self.nodeWeightList[
                vertex0] - self.nodeWeightedInverseLaplacian[:, vertex1] / self.nodeWeightList[vertex1]
            vTempDotInv = self.nodeWeightedInverseLaplacian[vertex0] - self.nodeWeightedInverseLaplacian[vertex1]
            effectiveResistanceOut[index] = invDotUTemp[vertex0] - invDotUTemp[vertex1]
            edgeImportanceOut[index] = np.dot(invDotUTemp, vTempDotInv)
            neighbors0 = [indexInner for indexInner, item in enumerate(self.adjacency[vertex0]) if item > 0]
            neighbors1 = [indexInner for indexInner, item in enumerate(self.adjacency[vertex1]) if item > 0]
            numTrianglesOut[index] = len([item for item in neighbors0 if item in neighbors1])

        endTime = time.time()
        if diagnosticSwitch > 1: print('make_wOmega_m_tau: ', endTime - startTime)
        return [edgesToSample, effectiveResistanceOut * self.edgeWeightList[edgesToSample],
                edgeImportanceOut * self.edgeWeightList[edgesToSample], numTrianglesOut]

    def wOmega_m_to_betaStar(self, wOmegaIn, mIn, tauIn, pMin=0.125, reductionType='both', reductionTarget='edges',
                             maxReweightFactor=0):
        startTime = time.time()
        if reductionType == 'delete' and reductionTarget == 'nodes':
            print('Cannot do deletion only when targeting reduction of nodes')
            return
        if wOmegaIn < -1.0e-12 or wOmegaIn > 1.0 + 1.0e-12:
            print("ERROR IN WR")
        if reductionTarget == 'edges':
            if reductionType == 'delete':
                if wOmegaIn > 1.0 - 10e-6:
                    return [0.0, [0.0, 0.0, 1.0, 1.0]]
                minBetaStarTemp = mIn / (1 - wOmegaIn) / (1 - pMin)
                deletionProbTemp = pMin
                contractionProbTemp = 0.0
                reweightProbTemp = 1.0 - pMin
                reweightFactorTemp = (1.0 - deletionProbTemp / (1.0 - wOmegaIn)) ** -1
                if maxReweightFactor > 0:
                    if deletionProbTemp > (1.0 - maxReweightFactor ** -1) * (1.0 - wOmegaIn):
                        deletionProbTemp = (1.0 - maxReweightFactor ** -1) * (1.0 - wOmegaIn)
                        reweightProbTemp = 1.0 - deletionProbTemp
                        minBetaStarTemp = mIn / (1 - wOmegaIn) / (1 - deletionProbTemp)
                        reweightFactorTemp = (1.0 - deletionProbTemp / (1.0 - wOmegaIn)) ** -1
                actionProbReweightTemp = [deletionProbTemp, contractionProbTemp, reweightProbTemp, reweightFactorTemp]
            elif reductionType == 'contract':
                minBetaStarTemp = mIn / wOmegaIn / (1.0 - pMin) / (1.0 + tauIn) ** 0.5
                deletionProbTemp = 0.0
                contractionProbTemp = pMin
                reweightProbTemp = 1.0 - pMin
                reweightFactorTemp = 1.0 - contractionProbTemp / wOmegaIn
                if contractionProbTemp > wOmegaIn:
                    minBetaStarTemp = mIn / wOmegaIn / (1.0 - wOmegaIn) / (1 + (1.0 + tauIn) ** 0.5)
                    deletionProbTemp = 1.0 - wOmegaIn
                    contractionProbTemp = wOmegaIn
                    reweightProbTemp = 0.0
                    reweightFactorTemp = 1.0
                actionProbReweightTemp = [deletionProbTemp, contractionProbTemp, reweightProbTemp, reweightFactorTemp]
            elif reductionType == 'both':
                if wOmegaIn > 1.0 - 10e-14:
                    minBetaStarTemp = mIn / wOmegaIn / (1.0 - pMin) / (1.0 + tauIn) ** 0.5
                    deletionProbTemp = 0.0
                    contractionProbTemp = pMin
                    reweightProbTemp = 1.0 - pMin
                    reweightFactorTemp = 1.0 - contractionProbTemp / wOmegaIn
                    if maxReweightFactor > 0:
                        if reweightFactorTemp < maxReweightFactor ** -1:
                            contractionProbTemp = (1.0 - maxReweightFactor ** -1) * (wOmegaIn)
                            reweightProbTemp = 1.0 - contractionProbTemp
                            minBetaStarTemp = mIn / wOmegaIn / (1.0 - contractionProbTemp) / (1.0 + tauIn) ** 0.5
                            reweightFactorTemp = 1.0 - contractionProbTemp / wOmegaIn
                    actionProbReweightTemp = [deletionProbTemp, contractionProbTemp, reweightProbTemp,
                                              reweightFactorTemp]
                else:
                    minBetaStarTempList = [mIn / (1.0 - wOmegaIn) / (1.0 - pMin),
                                           mIn / wOmegaIn / (1.0 - pMin) / (1.0 + tauIn) ** 0.5]
                    minBetaStarIndex = np.argmin(minBetaStarTempList)
                    if minBetaStarIndex == 0 and minBetaStarTempList[0] != minBetaStarTempList[1]:
                        minBetaStarTemp = minBetaStarTempList[0]
                        deletionProbTemp = pMin
                        contractionProbTemp = 0.0
                        reweightProbTemp = 1.0 - pMin
                        reweightFactorTemp = (1.0 - deletionProbTemp / (1.0 - wOmegaIn)) ** -1
                    else:
                        minBetaStarTemp = minBetaStarTempList[1]
                        deletionProbTemp = 0.0
                        contractionProbTemp = pMin
                        reweightProbTemp = 1.0 - pMin
                        reweightFactorTemp = 1.0 - contractionProbTemp / wOmegaIn
                    if contractionProbTemp > wOmegaIn:
                        minBetaStarTemp = mIn / wOmegaIn / (1.0 - wOmegaIn) / (1 + (1.0 + tauIn) ** 0.5)
                        deletionProbTemp = 1.0 - wOmegaIn
                        contractionProbTemp = wOmegaIn
                        reweightProbTemp = 0.0
                        reweightFactorTemp = 1.0
                    if deletionProbTemp > 1.0 - wOmegaIn:
                        minBetaStarTemp = mIn / wOmegaIn / (1.0 - wOmegaIn) / (1 + (1.0 + tauIn) ** 0.5)
                        deletionProbTemp = 1.0 - wOmegaIn
                        contractionProbTemp = wOmegaIn
                        reweightProbTemp = 0.0
                        reweightFactorTemp = 1.0
                    actionProbReweightTemp = [deletionProbTemp, contractionProbTemp, reweightProbTemp,
                                              reweightFactorTemp]

        if reductionTarget == 'nodes':
            minBetaStarTemp = mIn / wOmegaIn / (1.0 - pMin)
            deletionProbTemp = 0.0
            contractionProbTemp = pMin
            reweightProbTemp = 1.0 - pMin
            reweightFactorTemp = 1.0 - contractionProbTemp / wOmegaIn
            if contractionProbTemp > wOmegaIn:
                minBetaStarTemp = mIn / wOmegaIn / (1.0 - wOmegaIn)
                deletionProbTemp = 1.0 - wOmegaIn
                contractionProbTemp = wOmegaIn
                reweightProbTemp = 0.0
                reweightFactorTemp = 1.0
            actionProbReweightTemp = [deletionProbTemp, contractionProbTemp, reweightProbTemp, reweightFactorTemp]

        endTime = time.time()
        if diagnosticSwitch > 1: print('wOmega_m_to_betaStar: ', endTime - startTime)
        return minBetaStarTemp, actionProbReweightTemp

    def wOmega_m_to_betaStarList(self, wOmegaListIn, mListIn, tauListIn, pMin=0.125, reductionType='both',
                                 reductionTarget='edges', maxReweightFactor=0):
        startTime = time.time()
        minBetaStarListOut = np.zeros(len(wOmegaListIn))
        actionProbReweightListOut = np.zeros((len(wOmegaListIn), 4))
        for index in range(len(wOmegaListIn)):
            minBetaStarTemp, actionProbReweightTemp = self.wOmega_m_to_betaStar(wOmegaListIn[index], mListIn[index],
                                                                                tauListIn[index], pMin=pMin,
                                                                                reductionType=reductionType,
                                                                                reductionTarget=reductionTarget,
                                                                                maxReweightFactor=maxReweightFactor)
            minBetaStarListOut[index] = minBetaStarTemp
            actionProbReweightListOut[index] = actionProbReweightTemp
        endTime = time.time()
        if diagnosticSwitch > 1: print('wOmega_m_to_betaStarList: ', endTime - startTime)
        return minBetaStarListOut, actionProbReweightListOut

    def reduce_graph_single_edge(self, numSamples=1, pMin=0.125, reductionType='both', reductionTarget='edges',
                                 maxReweightFactor=0):
        startTime = time.time()
        if not self.updatedInverses:
            self.update_inverse_laplacian()
        sampledEdgeList, sampledWOmegaList, sampledMList, sampledTauList = self.make_wOmega_m_tau(method='random',
                                                                                                  numSamples=numSamples)
        sampledMinBetaStarList, sampledActionProbReweightList = self.wOmega_m_to_betaStarList(sampledWOmegaList,
                                                                                              sampledMList,
                                                                                              sampledTauList, pMin=pMin,
                                                                                              reductionType=reductionType,
                                                                                              reductionTarget=reductionTarget,
                                                                                              maxReweightFactor=maxReweightFactor)
        nonzeroIndices = [index for index, item in enumerate(sampledActionProbReweightList) if
                          not (item[0] == 0.0 and item[1] == 0.0)]
        if len(nonzeroIndices) == 0: return
        chosenEdgeIndex = nonzeroIndices[np.argmin(sampledMinBetaStarList[nonzeroIndices])]

        chosenEdgeRealIndex = sampledEdgeList[chosenEdgeIndex]
        chosenActionProbReweight = sampledActionProbReweightList[chosenEdgeIndex]
        edgeActionProbs = chosenActionProbReweight[0:3]
        edgeAction = np.random.choice(range(3), p=edgeActionProbs)

        if edgeAction == 0:
            if diagnosticSwitch > 0: print('deleting edge ', self.edgeList[chosenEdgeRealIndex])
            self.delete_edge(chosenEdgeRealIndex)
        if edgeAction == 1:
            if diagnosticSwitch > 0: print('contracting edge ', self.edgeList[chosenEdgeRealIndex])
            self.contract_edge(chosenEdgeRealIndex)
        if edgeAction == 2 and chosenActionProbReweight[3] != 1.0:
            if diagnosticSwitch > 0: print('reweighting edge ', self.edgeList[chosenEdgeRealIndex], ' by factor ',
                                           chosenActionProbReweight[3])
            self.reweight_edge(chosenEdgeRealIndex, chosenActionProbReweight[3])

        endTime = time.time()
        if diagnosticSwitch > 1: print('reduce_graph_single_edge: ', endTime - startTime)

    def delete_edge(self, edgeIndexIn):
        startTime = time.time()
        changeTemp = -1.0 * self.edgeWeightList[edgeIndexIn]
        nodesTemp = self.edgeList[edgeIndexIn]
        self.adjacency[nodesTemp[0], nodesTemp[1]] = 0.0
        self.adjacency[nodesTemp[1], nodesTemp[0]] = 0.0
        self.laplacian = self.adjacency_to_laplacian(self.adjacency)
        self.nodeWeightedLaplacian = (((self.laplacian).T) / self.nodeWeightList).T
        self.edgeList = np.delete(self.edgeList, edgeIndexIn, 0)
        self.edgeWeightList = np.delete(self.edgeWeightList, edgeIndexIn, 0)

        self.updatedInverses = False
        (self.updateList).append([nodesTemp, 1.0 / changeTemp])
        endTime = time.time()
        if diagnosticSwitch > 1: print('delete_edge: ', endTime - startTime)

    def reweight_edge(self, edgeIndexIn, reweightFactorIn):
        startTime = time.time()
        changeTemp = (reweightFactorIn - 1.0) * self.edgeWeightList[edgeIndexIn]
        nodesTemp = self.edgeList[edgeIndexIn]
        self.adjacency[nodesTemp[0], nodesTemp[1]] += changeTemp
        self.adjacency[nodesTemp[1], nodesTemp[0]] += changeTemp
        self.laplacian = self.adjacency_to_laplacian(self.adjacency)
        self.nodeWeightedLaplacian = (((self.laplacian).T) / self.nodeWeightList).T
        self.edgeWeightList[edgeIndexIn] += changeTemp

        self.updatedInverses = False
        (self.updateList).append([nodesTemp, 1.0 / changeTemp])
        endTime = time.time()
        if diagnosticSwitch > 1: print('reweight_edge: ', endTime - startTime)

    def contract_edge(self, edgeIndexIn):
        startTime = time.time()
        nodesToContract = [int(self.edgeList[int(edgeIndexIn), 0]), int(self.edgeList[int(edgeIndexIn), 1])]
        edgeWeightToContract = self.edgeWeightList[edgeIndexIn]
        layoutTemp = self.layout
        tempElementLayoutTemp = np.array(
            [(layoutTemp[nodesToContract[0]][index] * self.nodeWeightList[nodesToContract[0]] \
              + layoutTemp[nodesToContract[1]][index] * self.nodeWeightList[nodesToContract[1]]) \
             for index in range(len(layoutTemp[nodesToContract[0]]))]) \
                                / (self.nodeWeightList[nodesToContract[0]] + self.nodeWeightList[nodesToContract[1]])
        layoutTemp[nodesToContract[0]] = tuple(tempElementLayoutTemp)
        if nodesToContract[1] == 0:
            layoutTemp = layoutTemp[(nodesToContract[1] + 1):]
        elif nodesToContract[1] == len(layoutTemp) - 1:
            layoutTemp = layoutTemp[0:nodesToContract[1]]
        else:
            layoutTemp = np.concatenate((layoutTemp[0:nodesToContract[1]], layoutTemp[(nodesToContract[1] + 1):]))
        self.layout = layoutTemp

        # self.nodeWeightListOld = np.copy(self.nodeWeightList)

        self.contractedNodesToNodes[:, nodesToContract[0]] += self.contractedNodesToNodes[:, nodesToContract[1]]
        self.contractedNodesToNodes = (np.delete(self.contractedNodesToNodes.T, nodesToContract[1], 0)).T

        self.nodeList = np.delete(self.nodeList, nodesToContract[1], 0)
        self.nodeWeightList = np.dot(self.contractedNodesToNodes.T, self.nodeWeightsIn)

        self.adjacency[nodesToContract[0], nodesToContract[1]] = 0.0
        self.adjacency[nodesToContract[1], nodesToContract[0]] = 0.0
        self.adjacency[nodesToContract[0], :] += self.adjacency[nodesToContract[1], :]
        self.adjacency[:, nodesToContract[0]] += self.adjacency[:, nodesToContract[1]]
        self.adjacency = np.delete(self.adjacency, nodesToContract[1], 0)
        self.adjacency = (np.delete(self.adjacency.T, nodesToContract[1], 0)).T

        edgeListTemp = []
        edgeWeightListTemp = []
        for i in range(len(self.adjacency)):
            for j in range(i, len(self.adjacency)):
                if self.adjacency[i, j] > 0:
                    edgeListTemp.append([i, j])
                    edgeWeightListTemp.append(self.adjacency[i, j])

        self.edgeList = np.array(edgeListTemp)
        self.edgeWeightList = np.array(edgeWeightListTemp)

        self.laplacian = self.adjacency_to_laplacian(self.adjacency)
        self.nodeWeightedLaplacian = (((self.laplacian).T) / self.nodeWeightList).T

        self.updatedInverses = False
        (self.updateList).append([nodesToContract, 0.0])
        (self.rowsToDelete).append(nodesToContract)
        endTime = time.time()
        if diagnosticSwitch > 1: print('contract_edge: ', endTime - startTime)

    def make_incidence_row(self, numTotalIn, edgeIn):
        rowOut = np.zeros(numTotalIn)
        rowOut[edgeIn[0]] = 1
        rowOut[edgeIn[1]] = -1
        return rowOut

    def update_inverse_laplacian(self):
        startTime = time.time()

        edgesToChange = [item[0] for item in self.updateList]
        inverseChange = [item[1] for item in self.updateList]

        incidenceTemp = np.array([self.make_incidence_row(len(self.nodeWeightListOld), edge) for edge in edgesToChange])

        uTemp = (incidenceTemp / self.nodeWeightListOld).T
        vTemp = incidenceTemp

        try:
            easierInverse = np.linalg.inv(
                np.diag(inverseChange) + np.dot(vTemp, np.dot(self.nodeWeightedInverseLaplacian, uTemp)))
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                self.thereIsAProblem = True
                print("Problem: singular matrix when updating Laplacian")
                return
            else:
                raise

        if np.shape(easierInverse) == (1, 1):
            invLapUpdate = -easierInverse[0, 0] * np.outer(np.dot(self.nodeWeightedInverseLaplacian, uTemp),
                                                           np.dot(vTemp, self.nodeWeightedInverseLaplacian))
        else:
            invLapUpdate = -np.dot(np.dot(np.dot(self.nodeWeightedInverseLaplacian, uTemp), easierInverse),
                                   np.dot(vTemp, self.nodeWeightedInverseLaplacian))

        self.nodeWeightedInverseLaplacian += invLapUpdate
        if len(self.rowsToDelete) > 0:
            for rowToDelete in self.rowsToDelete:
                self.nodeWeightedInverseLaplacian[:, rowToDelete[0]] += self.nodeWeightedInverseLaplacian[:,
                                                                        rowToDelete[1]]
                self.nodeWeightedInverseLaplacian = np.delete(self.nodeWeightedInverseLaplacian, rowToDelete[1], 0)
                self.nodeWeightedInverseLaplacian = (
                    np.delete(self.nodeWeightedInverseLaplacian.T, rowToDelete[1], 0)).T

        self.updatedInverses = True
        self.updateList = []
        self.rowsToDelete = []
        self.nodeWeightListOld = np.copy(self.nodeWeightList)

        endTime = time.time()
        if diagnosticSwitch > 1: print('update_inverse_laplacian, ', endTime - startTime)

    def get_edgeList_proposal_RM(self, numSamplesIn='all'):
        adjacencyTemp = self.adjacency
        edgeListTemp = list([list(item) for item in self.edgeList])
        randomNodeOrderTemp = np.random.permutation(len(adjacencyTemp))
        nodePairsOut = []
        matchedNodesTemp = []

        if numSamplesIn == 'all':
            numSamples = len(self.edgeList)
        else:
            numSamples = numSamplesIn

        for firstNode in randomNodeOrderTemp:
            if firstNode not in matchedNodesTemp:
                unmatchedNeighborsTemp = [index for index, item in enumerate(adjacencyTemp[firstNode]) if
                                          item > 0 and index not in matchedNodesTemp]
                if len(unmatchedNeighborsTemp) > 0:
                    secondNode = np.random.choice(unmatchedNeighborsTemp)
                    nodePairsOut.append(sorted([firstNode, secondNode]))
                    matchedNodesTemp.append(firstNode)
                    matchedNodesTemp.append(secondNode)
            if len(nodePairsOut) >= numSamples:
                break

        proposedEdgeListOut = [edgeListTemp.index(item) for item in nodePairsOut]
        return proposedEdgeListOut

    def reduce_graph_multi_edge(self, numSamples='all', qFraction=0.0625, pMin=0.125, reductionType='both',
                                reductionTarget='edges', maxReweightFactor=0):
        if not self.updatedInverses:
            self.update_inverse_laplacian()
        sampledEdgeList, sampledWOmegaList, sampledMList, sampledTauList = self.make_wOmega_m_tau(method='RM',
                                                                                                  numSamples=numSamples)
        sampledMinBetaStarList, sampledActionProbReweightList = self.wOmega_m_to_betaStarList(sampledWOmegaList,
                                                                                              sampledMList,
                                                                                              sampledTauList, pMin=pMin,
                                                                                              reductionType=reductionType,
                                                                                              reductionTarget=reductionTarget,
                                                                                              maxReweightFactor=maxReweightFactor)
        nonzeroIndices = [index for index, item in enumerate(sampledActionProbReweightList) if
                          not (item[0] == 0.0 and item[1] == 0.0)]
        if len(nonzeroIndices) == 0: return

        numPerturbationsTemp = np.max([1, int(round(qFraction * len(nonzeroIndices)))])
        chosenEdgesIndices = np.array(nonzeroIndices)[
            list(np.argsort(np.array(sampledMinBetaStarList)[nonzeroIndices])[:numPerturbationsTemp])]
        chosenEdgesRealIndices = np.array(sampledEdgeList)[chosenEdgesIndices]
        chosenActionProbReweightList = np.array(sampledActionProbReweightList)[chosenEdgesIndices]

        edgesToDelete = []
        edgesToContract = []
        for index, chosenEdgeRealIndex in enumerate(chosenEdgesRealIndices):

            edgeActionProbs = chosenActionProbReweightList[index][0:3]
            edgeAction = np.random.choice(range(3), p=edgeActionProbs)

            if edgeAction == 0:
                if diagnosticSwitch > 0: print('deleting edge ', chosenEdgeRealIndex)
                edgesToDelete.append(chosenEdgeRealIndex)
            if edgeAction == 1:
                if diagnosticSwitch > 0: print('contracting edge ', chosenEdgeRealIndex)
                edgesToContract.append(chosenEdgeRealIndex)
            if edgeAction == 2 and chosenActionProbReweightList[index][3] != 1.0:
                if diagnosticSwitch > 0: print('reweighting edge ', chosenEdgeRealIndex, ' by factor ',
                                               chosenActionProbReweightList[index][3])
                self.reweight_edge(chosenEdgeRealIndex, chosenActionProbReweightList[index][3])
        edgesToDelete = sorted(edgesToDelete)

        contractSwitch = True
        if edgesToContract == []:
            shiftedEdgesToContract = []
            contractSwitch = False
        else:
            shiftedEdgesToContract = [
                int(edgeToContract - len([item for item in edgesToDelete if edgeToContract > item])) for edgeToContract
                in edgesToContract]

        # self.nodeWeightListOld = np.copy(self.nodeWeightList)
        self.delete_multiple_edges(edgesToDelete)
        if contractSwitch: self.contract_multiple_edges(shiftedEdgesToContract)

    def delete_multiple_edges(self, edgeIndexListIn):
        startTime = time.time()
        for edgeIndex in edgeIndexListIn:
            changeTemp = -1.0 * self.edgeWeightList[edgeIndex]
            nodesTemp = self.edgeList[edgeIndex]
            self.adjacency[nodesTemp[0], nodesTemp[1]] = 0.0
            self.adjacency[nodesTemp[1], nodesTemp[0]] = 0.0
            (self.updateList).append([nodesTemp, 1.0 / changeTemp])

        self.laplacian = self.adjacency_to_laplacian(self.adjacency)
        self.nodeWeightedLaplacian = (((self.laplacian).T) / self.nodeWeightList).T
        self.edgeList = np.delete(self.edgeList, edgeIndexListIn, 0)
        self.edgeWeightList = np.delete(self.edgeWeightList, edgeIndexListIn, 0)

        self.updatedInverses = False
        endTime = time.time()
        if diagnosticSwitch > 1: print('delete_edge, ', endTime - startTime)

    def contract_multiple_edges(self, edgeIndexListIn):  # ONLY WORKS WITH EDGES THAT DON'T SHARE NODES!!!
        startContractTime = time.time()

        nodesToContract = np.array(
            [sorted([int(self.edgeList[int(edge), 0]), int(self.edgeList[int(edge), 1])]) for edge in edgeIndexListIn])
        edgeSortingArgs = np.argsort(-np.array(nodesToContract[:, 1]))

        sortedNodesToContract = [nodesToContract[index] for index in edgeSortingArgs]
        sortedEdgesToContract = [edgeIndexListIn[index] for index in edgeSortingArgs]

        edgeWeightListTemp = np.array([self.edgeWeightList[int(edge)] for edge in edgeIndexListIn])
        sortedEdgeWeightListTemp = [edgeWeightListTemp[index] for index in edgeSortingArgs]
        for index in range(len(edgeSortingArgs)):
            (self.updateList).append([sortedNodesToContract[index], 0.0])
            (self.rowsToDelete).append(sortedNodesToContract[index])

        for index, nodePair in enumerate(sortedNodesToContract):
            self.contract_nodePair(nodePair, sortedEdgeWeightListTemp[index])

    def contract_nodePair(self, nodePair, edgeWeightIn=1.0):
        startTime = time.time()
        nodesToContract = nodePair
        edgeWeightToContract = edgeWeightIn
        layoutTemp = self.layout
        tempElementLayoutTemp = np.array(
            [(layoutTemp[nodesToContract[0]][index] * self.nodeWeightList[nodesToContract[0]] \
              + layoutTemp[nodesToContract[1]][index] * self.nodeWeightList[nodesToContract[1]]) \
             for index in range(len(layoutTemp[nodesToContract[0]]))]) \
                                / (self.nodeWeightList[nodesToContract[0]] + self.nodeWeightList[nodesToContract[1]])
        layoutTemp[nodesToContract[0]] = tuple(tempElementLayoutTemp)
        if nodesToContract[1] == 0:
            layoutTemp = layoutTemp[(nodesToContract[1] + 1):]
        elif nodesToContract[1] == len(layoutTemp) - 1:
            layoutTemp = layoutTemp[0:nodesToContract[1]]
        else:
            layoutTemp = np.concatenate((layoutTemp[0:nodesToContract[1]], layoutTemp[(nodesToContract[1] + 1):]))
        self.layout = layoutTemp

        self.contractedNodesToNodes[:, nodesToContract[0]] += self.contractedNodesToNodes[:, nodesToContract[1]]
        self.contractedNodesToNodes = (np.delete(self.contractedNodesToNodes.T, nodesToContract[1], 0)).T

        self.nodeList = np.delete(self.nodeList, nodesToContract[1], 0)
        self.nodeWeightList = np.dot(self.contractedNodesToNodes.T, self.nodeWeightsIn)

        self.adjacency[nodesToContract[0], nodesToContract[1]] = 0.0
        self.adjacency[nodesToContract[1], nodesToContract[0]] = 0.0
        self.adjacency[nodesToContract[0], :] += self.adjacency[nodesToContract[1], :]
        self.adjacency[:, nodesToContract[0]] += self.adjacency[:, nodesToContract[1]]
        self.adjacency = np.delete(self.adjacency, nodesToContract[1], 0)
        self.adjacency = (np.delete(self.adjacency.T, nodesToContract[1], 0)).T

        edgeListTemp = []
        edgeWeightListTemp = []
        for i in range(len(self.adjacency)):
            for j in range(i, len(self.adjacency)):
                if self.adjacency[i, j] > 0:
                    edgeListTemp.append([i, j])
                    edgeWeightListTemp.append(self.adjacency[i, j])

        self.edgeList = np.array(edgeListTemp)
        self.edgeWeightList = np.array(edgeWeightListTemp)

        self.laplacian = self.adjacency_to_laplacian(self.adjacency)
        self.nodeWeightedLaplacian = (((self.laplacian).T) / self.nodeWeightList).T

        self.updatedInverses = False
        endTime = time.time()
        if diagnosticSwitch > 1: print('contract_nodePair, ', endTime - startTime)
