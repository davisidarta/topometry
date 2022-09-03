# Procrustes analysis
# This is from the Procrustes [package](https://github.com/theochem/procrustes)
# The generalized procrustes problem is implemented
# here separately to avoid an unecessary dependency.
# If you use this code, please cite:
# @article{Meng2022procrustes,
#     title = {Procrustes: A python library to find transformations that maximize the similarity between matrices},
#     author = {Fanwang Meng and Michael Richer and Alireza Tehrani and Jonathan La and Taewon David Kim and Paul W. Ayers and Farnaz Heidar-Zadeh},
#     journal = {Computer Physics Communications},
#     volume = {276},
#     number = {108334},
#     pages = {1--37},
#     year = {2022},
#     issn = {0010-4655},
#     doi = {https://doi.org/10.1016/j.cpc.2022.108334},
#     url = {https://www.sciencedirect.com/science/article/pii/S0010465522000522},
#     keywords = {Procrustes analysis, Orthogonal, Symmetric, Rotational, Permutation, Softassign},
# }

import numpy as np
from scipy.linalg import svd
from sklearn.base import BaseEstimator, TransformerMixin


def _zero_padding(
    array_a, array_b, pad_mode = "row-col"
):
    """
    This is from the Procrustes [package](https://github.com/theochem/procrustes)
    Return arrays padded with rows and/or columns of zero.
    Parameters
    ----------
    array_a : ndarray
        The 2D-array :math:`\mathbf{A}_{n_a \times m_a}`.
    array_b : ndarray
        The 2D-array :math:`\mathbf{B}_{n_b \times m_b}`.
    pad_mode : str
        Specifying how to pad the arrays. Should be one of
        - "row"
            The array with fewer rows is padded with zero rows so that both have the same
            number of rows.
        - "col"
            The array with fewer columns is padded with zero columns so that both have the
            same number of columns.
        - "row-col"
            The array with fewer rows is padded with zero rows, and the array with fewer
            columns is padded with zero columns, so that both have the same dimensions.
            This does not necessarily result in square arrays.
        - "square"
            The arrays are padded with zero rows and zero columns so that they are both
            squared arrays. The dimension of square array is specified based on the highest
            dimension, i.e. :math:`\text{max}(n_a, m_a, n_b, m_b)`.
    Returns
    -------
    padded_a : ndarray
        Padded array_a.
    padded_b : ndarray
        Padded array_b.
    """
    # sanity checks
    if not isinstance(array_a, np.ndarray) or not isinstance(array_b, np.ndarray):
        raise ValueError("Arguments array_a & array_b should be numpy arrays.")
    if array_a.ndim != 2 or array_b.ndim != 2:
        raise ValueError("Arguments array_a & array_b should be 2D arrays.")

    if array_a.shape == array_b.shape and array_a.shape[0] == array_a.shape[1]:
        # special case of square arrays, mode is set to None so that array_a & array_b are returned.
        pad_mode = None

    if pad_mode == "square":
        # calculate desired dimension of square array
        (a_n1, a_m1), (a_n2, a_m2) = array_a.shape, array_b.shape
        dim = max(a_n1, a_n2, a_m1, a_m2)
        # padding rows to have both arrays have dim rows
        if a_n1 < dim:
            array_a = np.pad(array_a, [[0, dim - a_n1], [0, 0]], "constant", constant_values=0)
        if a_n2 < dim:
            array_b = np.pad(array_b, [[0, dim - a_n2], [0, 0]], "constant", constant_values=0)
        # padding columns to have both arrays have dim columns
        if a_m1 < dim:
            array_a = np.pad(array_a, [[0, 0], [0, dim - a_m1]], "constant", constant_values=0)
        if a_m2 < dim:
            array_b = np.pad(array_b, [[0, 0], [0, dim - a_m2]], "constant", constant_values=0)

    if pad_mode in ["row", "row-col"]:
        # padding rows to have both arrays have the same number of rows
        diff = array_a.shape[0] - array_b.shape[0]
        if diff < 0:
            array_a = np.pad(array_a, [[0, -diff], [0, 0]], "constant", constant_values=0)
        else:
            array_b = np.pad(array_b, [[0, diff], [0, 0]], "constant", constant_values=0)

    if pad_mode in ["col", "row-col"]:
        # padding columns to have both arrays have the same number of columns
        diff = array_a.shape[1] - array_b.shape[1]
        if diff < 0:
            array_a = np.pad(array_a, [[0, 0], [0, -diff]], "constant", constant_values=0)
        else:
            array_b = np.pad(array_b, [[0, 0], [0, diff]], "constant", constant_values=0)

    return array_a, array_b


def _translate_array(
    array_a, array_b = None, weight = None
):
    """
    This is from the Procrustes [package](https://github.com/theochem/procrustes)
    Return translated array_a and translation vector.
    Columns of both arrays will have mean zero.
    Parameters
    ----------
    array_a : ndarray
        The 2D-array to translate.
    array_b : ndarray, optional
        The 2D-array to translate array_a based on.
    weight : ndarray, optional
        The weight vector.
    Returns
    -------
    array_a : ndarray
        If array_b is None, array_a is translated to origin using its centroid.
        If array_b is given, array_a is translated to centroid of array_b (the centroid of
        translated array_a will centroid with the centroid array_b).
    centroid : float
        If array_b is given, the centroid is returned.
    """
    # The mean is strongly affected by outliers and is not a robust estimator for central location
    # see https://docs.python.org/3.6/library/statistics.html?highlight=mean#statistics.mean
    if weight is not None:
        if weight.ndim != 1:
            raise ValueError("The weight should be a 1d row vector.")
        if not (weight >= 0).all():
            raise ValueError("The elements of the weight should be non-negative.")

    centroid_a = np.average(array_a, axis=0, weights=weight)
    if array_b is not None:
        # translation vector to b centroid
        centroid_a -= np.average(array_b, axis=0, weights=weight)
    return array_a - centroid_a, -1 * centroid_a


def _scale_array(array_a, array_b=None):
    """
    This is from the Procrustes [package](https://github.com/theochem/procrustes)
    Return scaled/normalized array_a and scaling vector.
    Parameters
    ----------
    array_a : ndarray
        The 2D-array to scale
    array_b : ndarray, default=None
        The 2D-array to scale array_a based on.
    Returns
    -------
    scaled_a, ndarray
        If array_b is None, array_a is normalized using the Frobenius norm.
        If array_b is given, array_a is scaled to match array_b"s norm (the norm of array_a
        will be equal norm of array_b).
    scale : float
        The scaling factor to match array_b norm.
    """
    # scaling factor to match unit sphere
    scale = 1.0 / np.linalg.norm(array_a)
    if array_b is not None:
        # scaling factor to match array_b norm
        scale *= np.linalg.norm(array_b)
    return array_a * scale, scale


def _hide_zero_padding(
    array_a,
    remove_zero_col = True,
    remove_zero_row = True,
    tol = 1.0e-8,
):
    """
    This is from the Procrustes [package](https://github.com/theochem/procrustes)
    Return array with zero-padded rows (bottom) and columns (right) removed.
    Parameters
    ----------
    array_a : ndarray
        The initial array.
    remove_zero_col : bool, optional
        If True, zero columns (values less than 1e-8) on the right side will be removed.
    remove_zero_row : bool, optional
        If True, zero rows (values less than 1e-8) on the bottom will be removed.
    tol : float, optional
        Tolerance value.
    Returns
    -------
    new_A : ndarray
        Array, with either near zero columns and/or zero rows are removed.
    """
    # Input checking
    if array_a.ndim > 2:
        raise TypeError("Matrix inputs must be 1- or 2- dimensional arrays")
    # Check zero rows from bottom to top
    if remove_zero_row:
        num_row = array_a.shape[0]
        tmp_a = array_a[..., np.newaxis] if array_a.ndim == 1 else array_a
        for array_v in tmp_a[::-1]:
            if any(abs(i) > tol for i in array_v):
                break
            num_row -= 1
        array_a = array_a[:num_row]
    # Cut off zero rows
    if remove_zero_col:
        if array_a.ndim == 2:
            # Check zero columns from right to left
            col_m = array_a.shape[1]
            for array_v in array_a.T[::-1]:
                if any(abs(i) > tol for i in array_v):
                    break
                col_m -= 1
            # Cut off zero columns
            array_a = array_a[:, :col_m]
    return array_a


def compute_error(
    a, b, t, s = None
):
    """
    This is from the Procrustes [package](https://github.com/theochem/procrustes).
    Return the one- or two-sided Procrustes (squared Frobenius norm) error.
    The double-sided Procrustes error is defined as
    .. math::
       \|\mathbf{S}\mathbf{A}\mathbf{T} - \mathbf{B}\|_{F}^2 =
       \text{Tr}\left[
            \left(\mathbf{S}\mathbf{A}\mathbf{T} - \mathbf{B}\right)^\dagger
            \left(\mathbf{S}\mathbf{A}\mathbf{T} - \mathbf{B}\right)\right]
    when :math:`\mathbf{S}` is the identity matrix :math:`\mathbf{I}`, this is called the one-sided
    Procrustes error.
    Parameters
    ----------
    a : ndarray
        The 2D-array :math:`\mathbf{A}_{m \times n}` which is going to be transformed.
    b : ndarray
        The 2D-array :math:`\mathbf{B}_{m \times n}` representing the reference matrix.
    t : ndarray
        The 2D-array :math:`\mathbf{T}_{n \times n}` representing the right-hand-side transformation
        matrix.
    s : ndarray, optional
        The 2D-array :math:`\mathbf{S}_{m \times m}` representing the left-hand-side transformation
        matrix. If set to `None`, the one-sided Procrustes error is computed.
    Returns
    -------
    error : float
        The squared Frobenius norm of difference between the transformed array, :math:`\mathbf{S}
        \mathbf{A}\mathbf{T}`, and the reference array, :math:`\mathbf{B}`.
    """
    # transform matrix A to either AT or SAT
    a_trans = np.dot(a, t) if s is None else np.dot(np.dot(s, a), t)
    # subtract matrix B and compute Frobenius norm squared
    return np.linalg.norm(a_trans - b, ord=None) ** 2


def setup_input_arrays(
    array_a,
    array_b,
    remove_zero_col,
    remove_zero_row,
    pad,
    translate,
    scale,
    check_finite,
    weight = None,
):
    """
    This is from the Procrustes [package](https://github.com/theochem/procrustes).
    Check and process array inputs for the Procrustes transformation routines.
    Usually, the precursor step before all Procrustes methods.
    Parameters
    ----------
    array_a : npdarray
        The 2D array :math:`A` being transformed.
    array_b : npdarray
        The 2D reference array :math:`B`.
    remove_zero_col : bool
        If True, zero columns (values less than 1e-8) on the right side will be removed.
    remove_zero_row : bool
        If True, zero rows (values less than 1e-8) on the bottom will be removed.
    pad : bool
        Add zero rows (at the bottom) and/or columns (to the right-hand side) of matrices
        :math:`\mathbf{A}` and :math:`\mathbf{B}` so that they have the same shape.
    translate : bool
        If true, then translate both arrays :math:`A, B` to the origin, ie columns of the arrays
        will have mean zero.
    scale :
        If True, both arrays are normalized to one with respect to the Frobenius norm, ie
        :math:`Tr(A^T A) = 1`.
    check_finite : bool
        If true, then checks if both arrays :math:`A, B` are numpy arrays and two-dimensional.
    weight : A list of ndarray or ndarray
        A list of the weight arrays or one numpy array. When only on numpy array provided,
        it is assumed that the two arrays :math:`A` and :math:`B` share the same weight matrix.
    Returns
    -------
    (ndarray, ndarray) :
        Returns the padded arrays, in that they have the same matrix dimensions.
    """
    array_a = _setup_input_array_lower(
        array_a, None, remove_zero_col, remove_zero_row, translate, scale, check_finite, weight
    )
    array_b = _setup_input_array_lower(
        array_b, None, remove_zero_col, remove_zero_row, translate, scale, check_finite, weight
    )
    if pad:
        array_a, array_b = _zero_padding(array_a, array_b, pad_mode="row-col")
    return array_a, array_b


def setup_input_arrays_multi(
    array_list,
    array_ref,
    remove_zero_col,
    remove_zero_row,
    pad_mode,
    translate,
    scale,
    check_finite,
    weight = None,
):
    """
    This is from the Procrustes [package](https://github.com/theochem/procrustes).
    Check and process array inputs for the Procrustes transformation routines.
    Parameters
    ----------
    array_list : List
        A list of 2D arrays that being transformed.
    array_ref : ndarray
        The 2D reference array :math:`B`.
    remove_zero_col : bool
        If True, zero columns (values less than 1e-8) on the right side will be removed.
    remove_zero_row : bool
        If True, zero rows (values less than 1e-8) on the bottom will be removed.
    pad_mode : str
        Specifying how to pad the arrays. Should be one of
            - "row"
                The array with fewer rows is padded with zero rows so that both have the same
                number of rows.
            - "col"
                The array with fewer columns is padded with zero columns so that both have the
                same number of columns.
            - "row-col"
                The array with fewer rows is padded with zero rows, and the array with fewer
                columns is padded with zero columns, so that both have the same dimensions.
                This does not necessarily result in square arrays.
            - "square"
                The arrays are padded with zero rows and zero columns so that they are both
                squared arrays. The dimension of square array is specified based on the highest
                dimension, i.e. :math:`\text{max}(n_a, m_a, n_b, m_b)`.
    translate : bool
        If true, then translate both arrays :math:`A, B` to the origin, ie columns of the arrays
        will have mean zero.
    scale :
        If True, both arrays are normalized to one with respect to the Frobenius norm, ie
        :math:`Tr(A^T A) = 1`.
    check_finite : bool
        If true, then checks if both arrays :math:`A, B` are numpy arrays and two-dimensional.
    weight : A list of ndarray or ndarray, optional
        A list of the weight arrays or one numpy array. When only on numpy array provided,
        it is assumed that the two arrays :math:`A` and :math:`B` share the same weight matrix.
    Returns
    -------
    List of arrays :
        Returns the padded arrays, in that they have the same matrix dimensions.
    """
    array_list_new = [
        _setup_input_array_lower(
            array_a=arr,
            array_ref=array_ref,
            remove_zero_col=remove_zero_col,
            remove_zero_row=remove_zero_row,
            translate=translate,
            scale=scale,
            check_finite=check_finite,
            weight=weight,
        )
        for arr in array_list
    ]
    arr_shape = np.array([arr.shape for arr in array_list_new])
    array_b = np.ones(np.max(arr_shape, axis=0), dtype=int)
    array_list_new = [_zero_padding(arr, array_b, pad_mode=pad_mode) for arr in array_list_new]
    return array_list_new


def _setup_input_array_lower(
    array_a,
    array_ref,
    remove_zero_col,
    remove_zero_row,
    translate,
    scale,
    check_finite,
    weight = None,
):
    """Pre-processing the matrices with translation, scaling."""
    _check_arraytypes(array_a)
    if check_finite:
        array_a = np.asarray_chkfinite(array_a)
        # Sometimes arrays already have zero padding that messes up zero padding below.
    array_a = _hide_zero_padding(array_a, remove_zero_col, remove_zero_row)
    if translate:
        array_a, _ = _translate_array(array_a, array_ref, weight)
    # scale the matrix when translate is False, but weight is True
    else:
        if weight is not None:
            array_a = np.dot(np.diag(weight), array_a)

    if scale:
        array_a, _ = _scale_array(array_a, array_ref)
    return array_a


def _check_arraytypes(*args):
    r"""Check array input types to Procrustes transformation routines."""
    if any(not isinstance(arr_x, np.ndarray) for arr_x in args):
        raise TypeError("Matrix inputs must be NumPy arrays")
    if any(x.ndim != 2 for x in args):
        raise TypeError("Matrix inputs must be 2-dimensional arrays")


class ProcrustesResult(dict):
    """
    This is from the Procrustes [package](https://github.com/theochem/procrustes).
    Represents the Procrustes analysis result.
    Attributes
    ----------
    error : float
        The Procrustes (squared Frobenius norm) error.
    new_a : ndarray
        The translated/scaled numpy ndarray :math:`\mathbf{A}`.
    new_b : ndarray
        The translated/scaled numpy ndarray :math:`\mathbf{B}`.
    t : ndarray
        The 2D-array :math:`\mathbf{T}` representing the right-hand-side transformation matrix.
    s : ndarray
        The 2D-array :math:`\mathbf{S}` representing the left-hand-side transformation
        matrix. If set to `None`, the one-sided Procrustes was performed.
    """

    # modification on https://github.com/scipy/scipy/blob/v1.4.1/scipy/optimize/optimize.py#L77-L132
    def __getattr__(self, name):
        """Deal with attributes which it doesn't explicitly manage."""
        try:
            return self[name]
        # Not using raise from makes the traceback inaccurate, because the message implies there
        # is a bug in the exception-handling code itself, which is a separate situation than
        # wrapping an exception
        # W0707 from http://pylint.pycqa.org/en/latest/technical_reference/features.html
        except KeyError as ke_info:
            raise AttributeError(name) from ke_info

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        """Return a human friendly representation."""
        if self.keys():
            max_len = max(map(len, list(self.keys()))) + 1
            return "\n".join([k.rjust(max_len) + ": " + repr(v) for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        """Provide basic customization of module attribute access with a list."""
        return list(self.keys())

def orthogonal(
    a,
    b,
    pad = True,
    translate = False,
    scale = False,
    unpad_col = False,
    unpad_row = False,
    check_finite = True,
    weight = None,
    lapack_driver = "gesvd"):
    """
    Perform orthogonal Procrustes.
    This is from the Procrustes [package](https://github.com/theochem/procrustes).

    Parameters
    ----------
    a : ndarray
        The 2D-array which is going to be transformed.
    b : ndarray
        The 2D-array representing the reference matrix.
    pad : bool, optional
        Add zero rows (at the bottom) and/or columns (to the right-hand side) of the matrices
     so that they have the same shape.
    translate : bool, optional
        If True, both arrays are centered at origin (columns of the arrays will have mean zero).
    scale : bool, optional
        If True, both arrays are normalized with respect to the Frobenius norm, i.e.,
    unpad_col : bool, optional
        If True, zero columns (with values less than 1.0e-8) on the right-hand side of the intial
         and  matrices are removed.
    unpad_row : bool, optional
        If True, zero rows (with values less than 1.0e-8) at the bottom of the intial
         and ` matrices are removed.
    check_finite : bool, optional
        If True, convert the input to an array, checking for NaNs or Infs.
    weight : ndarray, optional
        The 1D-array representing the weights of each row of. This defines the
        elements of the diagonal matrix  that is multiplied by 
        matrix.
    lapack_driver : {'gesvd', 'gesdd'}, optional
        Whether to use the more efficient divide-and-conquer approach ('gesdd') or the more robust
        general rectangular approach ('gesvd') to compute the singular-value decomposition with
        `scipy.linalg.svd`.
    Returns
    -------
    res : ProcrustesResult
        The Procrustes result represented as a class:`utils.ProcrustesResult` object.

    """
    # check inputs
    new_a, new_b = setup_input_arrays(
        a,
        b,
        unpad_col,
        unpad_row,
        pad,
        translate,
        scale,
        check_finite,
        weight,
    )
    if new_a.shape != new_b.shape:
        raise ValueError(
            f"Shape of A and B does not match: {new_a.shape} != {new_b.shape} "
            "Check pad, unpad_col, and unpad_row arguments."
        )
    # calculate SVD of A.T * B
    u, _, vt = svd(np.dot(new_a.T, new_b), lapack_driver=lapack_driver)
    # compute optimal orthogonal transformation
    u_opt = np.dot(u, vt)
    # compute one-sided error
    error = compute_error(new_a, new_b, u_opt)

    return ProcrustesResult(error=error, new_a=new_a, new_b=new_b, t=u_opt, s=None)


def generalized(
    array_list,
    ref = None,
    tol = 1.0e-7,
    n_iter = 200,
    check_finite = True,
):
    """
    Generalized Procrustes Analysis.
    This is from the Procrustes [package](https://github.com/theochem/procrustes).
    Parameters
    ----------
    array_list : List
        The list of 2D-array which is going to be transformed.
    ref : ndarray, optional
        The reference array to initialize the first iteration. If None, the first array in
        `array_list` will be used.
    tol: float, optional
        Tolerance value to stop the iterations.
    n_iter: int, optional
        Number of total iterations.
    check_finite : bool, optional
        If true, convert the input to an array, checking for NaNs or Infs.
    Returns
    -------
    array_aligned : List
        A list of transformed arrays with generalized Procrustes analysis.
    new_distance_gpa: float
        The distance for matching all the transformed arrays with generalized Procrustes analysis.
    Notes
    -----
    Given a set of matrices, :math:`\mathbf{A}_1, \mathbf{A}_2, \cdots, \mathbf{A}_k` with
    :math:`k > 2`,  the objective is to minimize in order to superimpose pairs of matrices.
    .. math::
        \min \quad = \sum_{i<j}^{j} {\left\| \mathbf{A}_i \mathbf{T}_i  -
         \mathbf{A}_j \mathbf{T}_j \right\| }^2
    This function implements the Equation (20) and the corresponding algorithm in  Gower's paper.
    """
    # check input arrays
    _check_arraytypes(*array_list)
    # check finite
    if check_finite:
        array_list = [np.asarray_chkfinite(arr) for arr in array_list]

    # todo: translation and scaling
    if n_iter <= 0:
        raise ValueError("Number of iterations should be a positive number.")
    if ref is None:
        # the first array will be used to build the initial ref
        array_aligned = [array_list[0]] + [
            _orthogonal(arr, array_list[0]) for arr in array_list[1:]
        ]
        ref = np.mean(array_aligned, axis=0)
    else:
        array_aligned = [None] * len(array_list)
        ref = ref.copy()

    distance_gpa = np.inf
    for _ in np.arange(n_iter):
        # align to ref
        array_aligned = [_orthogonal(arr, ref) for arr in array_list]
        # the mean
        new_ref = np.mean(array_aligned, axis=0)
        # todo: double check if the error is defined in the right way
        # the error
        new_distance_gpa = np.square(ref - new_ref).sum()
        if distance_gpa != np.inf and np.abs(new_distance_gpa - distance_gpa) < tol:
            break
        distance_gpa = new_distance_gpa
    return array_aligned, new_distance_gpa


class GeneralizedProcrustes(BaseEstimator, TransformerMixin):
    """
    Generalized Procrustes Analysis in a scikit-learn flavor.
    Automatically tries to align the provided matrices by finding transformations
    that make them as similar as possible to each other. This is from the Procrustes [package](https://github.com/theochem/procrustes),
    available under the GPL v3.
    
    Parameters
    ----------
    ref : ndarray, optional
        The reference array to initialize the first iteration. If None, the first array in
        `array_list` will be used.

    tol: float, optional
        Tolerance value to stop the iterations.

    n_iter: int, optional
        Number of total iterations.

    check_finite : bool, optional
        If true, convert the input to an array, checking for NaNs or Infs.

    """
    def __init__(self, ref = None, tol=1.0e-7, n_iter=200, check_finite=True):
        self.ref = ref
        self.tol = tol
        self.n_iter = n_iter
        self.check_finite = check_finite
        self.array_aligned_ = None
        self.new_distance_gpa_ = None

    def fit(self, array_list):
        """
        Fit the model with the given array_list.
        Parameters
        ----------
        array_list : list
            The list of 2D-array which is going to be transformed.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.array_aligned_, self.new_distance_gpa_ = generalized(
            array_list,
            self.ref,
            self.tol,
            self.n_iter,
            self.check_finite,
        )
        return self
    
    def transform(self, array_list=None):
        """
        Returns a tuple of the aligned concatenated array and the error (distance) for the matching. Here only for scikit-learn consistency.
        
        Parameters
        ----------
        array_list : List
            The list of 2D-array which is going to be transformed.
            
        Returns
        -------
        array_aligned : List
            A list of transformed arrays with generalized Procrustes analysis.

        new_distance_gpa: float
            The distance for matching all the transformed arrays with generalized Procrustes analysis.
        """
        if self.array_aligned_ is None:
            raise ValueError("The model has not been fitted yet. Call 'fit' first.")
        return self.array_aligned_, self.new_distance_gpa_

    def fit_transform(self, array_list):
        """
        Fit the model with the given array_list and returns a tuple of the aligned concatenated array and the error (distance) for the matching.

        Returns
        -------
        array_aligned : List
            A list of transformed arrays with generalized Procrustes analysis.

        new_distance_gpa: float
            The distance for matching all the transformed arrays with generalized Procrustes analysis.
        """
        self.fit(array_list)
        return self.array_aligned_, self.new_distance_gpa_

def _orthogonal(arr_a, arr_b):
    """Orthogonal Procrustes transformation and returns the transformed array."""
    res = orthogonal(arr_a, arr_b, translate=False, scale=False, unpad_col=False, unpad_row=False)
    return np.dot(res["new_a"], res["t"])


def fit_transform_procrustes(x, fit_transform_call, procrustes_batch_size=5000, procrustes_lm=1000):
    """
    Fit model and transform data for larger datasets. This is from GRAE (https://github.com/KevinMoonLab/GRAE).
    If dataset has more than self.proc_threshold samples, then compute the eigendecomposition or projection over
    mini-batches. In each batch, add self.procrustes_lm samples (which are the same for all batches),
    which can be used to compute a  procrustes transform to roughly align all batches in a coherent manner.

    Parameters
    ----------
    x: np.array
        Data to be transformed
    fit_transform_call: function
        Function to be called to fit and transform the data (scikit-learn style estimator).
    procrustes_batch_size: int
        Number of samples in each batch of procrustes
    procrustes_lm: int
        Number of anchor points present in all batches. Used as a reference for the procrustes
        transform.


    Returns:
    --------
    x_transformed: np.array
        Embedding of x, which is the union of all batches aligned with procrustes.
    """
    lm_points = x[:procrustes_lm, :]  # Reference points included in all batches
    initial_embedding = fit_transform_call(lm_points)
    result = [initial_embedding]
    remaining_x = x[procrustes_lm:, :]
    while len(remaining_x) != 0:
        if len(remaining_x) >= procrustes_batch_size:
            new_points = remaining_x[:procrustes_batch_size, :]
            remaining_x = np.delete(remaining_x,
                                    np.arange(procrustes_batch_size),
                                    axis=0)
        else:
            new_points = remaining_x
            remaining_x = np.delete(remaining_x,
                                    np.arange(len(remaining_x)),
                                    axis=0)

        subsetx = np.vstack((lm_points, new_points))
        subset_embedding = fit_transform_call(subsetx)

        d, Z, tform = procrustes(initial_embedding,
                                 subset_embedding[:procrustes_lm, :])

        subset_embedding_transformed = np.dot(
            subset_embedding[procrustes_lm:, :],
            tform['rotation']) + tform['translation']

        result.append(subset_embedding_transformed)
    return np.vstack(result)


def procrustes(X, Y, scaling=True, reflection='best'):

    """
    This is from GRAE (https://github.com/KevinMoonLab/GRAE).
    Taken from https://stackoverrun.com/es/q/5162566 adaptation of MATLAB to numpy.
    A port of MATLAB's `procrustes` function to Numpy.
    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.
        d, Z, [tform] = procrustes(X, Y)
    Inputs:
    ------------
    X, Y    
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.
    scaling 
        if False, the scaling component of the transformation is forced
        to 1
    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.
    Outputs
    ------------
    d       
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()
    Z
        the matrix of transformed Y-values
    tform   
        a dict specifying the rotation, translation and scaling that
        maps X --> Y
    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    #transformation values
    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform