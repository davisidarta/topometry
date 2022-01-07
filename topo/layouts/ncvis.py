# This script calls [NCVis](https://arxiv.org/abs/2001.11411) functions to operate on the TopOGraph class
# NCVis was developed by Aleksandr Artemenkov and Maxim Panov, and implemented at https://github.com/stat-ml/ncvis.
# This is rather a modular adaptation, and does not the optimization algorithm.
# NCVis is distributed with the following MIT license:
#
# MIT License
#
# Copyright (c) 2019-2020 Aleksandr Artemenkov
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


def NCVis(data,
          n_components=2,
          n_jobs=-1,
          n_neighbors=15,
          distance="cosine",
          M=15,
          efC=30,
          random_seed=42,
          n_epochs=50,
          n_init_epochs=20,
          spread=1.0,
          min_dist=0.4,
          alpha=1.0,
          a=None,
          b=None,
          alpha_Q=1.,
          n_noise=None,
          ):
    """
        Runs Noise Contrastive Visualization ((NCVis)[https://dl.acm.org/doi/abs/10.1145/3366423.3380061])
        for dimensionality reduction and graph layout .

        Parameters
        ----------
        n_components : int
            Desired dimensionality of the embedding.
        n_jobs : int
            The maximum number of threads to use. In case n_threads < 1, it defaults to the number of available CPUs.
        n_neighbors : int
            Number of nearest neighbours in the high dimensional space to consider.
        M : int
            The number of bi-directional links created for every new element during construction of HNSW.
            See https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
        efC : int
            The size of the dynamic list for the nearest neighbors (used during the search) in HNSW.
            See https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
        random_seed : int
            Random seed to initialize the generators. Notice, however, that the result may still depend on the number of threads.
        n_epochs : int
            The total number of epochs to run. During one epoch the positions of each nearest neighbors pair are updated.
        n_init_epochs : int
            The number of epochs used for initialization. During one epoch the positions of each nearest neighbors pair are updated.
        spread : float
            The effective scale of embedded points. In combination with ``min_dist``
            this determines how clustered/clumped the embedded points are.
            See https://github.com/lmcinnes/umap/blob/834184f9c0455f26db13ab148c0abd2d3767d968/umap/umap_.py#L1143
        min_dist : float
            The effective minimum distance between embedded points. Smaller values
            will result in a more clustered/clumped embedding where nearby points
            on the manifold are drawn closer together, while larger values will
            result on a more even dispersal of points. The value should be set
            relative to the ``spread`` value, which determines the scale at which
            embedded points will be spread out.
            See https://github.com/lmcinnes/umap/blob/834184f9c0455f26db13ab148c0abd2d3767d968/umap/umap_.py#L1135
        a : (optional, default None)
            More specific parameters controlling the embedding. If None these values
            are set automatically as determined by ``min_dist`` and ``spread``.
            See https://github.com/lmcinnes/umap/blob/834184f9c0455f26db13ab148c0abd2d3767d968/umap/umap_.py#L1179
        b : (optional, default None)
            More specific parameters controlling the embedding. If None these values
            are set automatically as determined by ``min_dist`` and ``spread``.
            See https://github.com/lmcinnes/umap/blob/834184f9c0455f26db13ab148c0abd2d3767d968/umap/umap_.py#L1183
        alpha : float
            Learning rate for the embedding positions.
        alpha_Q : float
            Learning rate for the normalization constant.
        n_noise : int or ndarray of ints
            Number of noise samples to use per data sample. If ndarray is provided, n_epochs is set to its length.
             If n_noise is None, it is set to dynamic sampling with noise level gradually increasing
              from 0 to fixed value.
        distance : str {'euclidean', 'cosine', 'correlation', 'inner_product'}
            Distance to use for nearest neighbors search.

    """

    try:
        import ncvis
    except ImportError('NCVis is needed for this embedding. Install it with `pip install ncvis`'):
        return print('NCVis is needed for this embedding. Install it with `pip install ncvis`')

    ncvis_emb = ncvis.NCVis(d=n_components,
                             n_threads=n_jobs,
                             n_neighbors=n_neighbors,
                             M=M,
                             ef_construction=efC,
                             random_seed=random_seed,
                             n_epochs=n_epochs,
                             n_init_epochs=n_init_epochs,
                             spread=spread,
                             min_dist=min_dist,
                             a=a,
                             b=b,
                             alpha=alpha,
                             alpha_Q=alpha_Q,
                             n_noise=n_noise,
                             distance=distance).fit_transform(data)

    return ncvis_emb



