import numpy as np

from utils.logger import get_logger


logger = get_logger()


def is_square_matrix(mat: np.ndarray):
    return mat.shape[0] == mat.shape[1]


def is_dimensions_same(mat1: np.ndarray, mat2: np.ndarray):
    return mat1.shape == mat2.shape


def affinity_matrix_refinement(affinity_mat, propagated_mat):
    assert is_square_matrix(affinity_mat)
    assert is_square_matrix(propagated_mat)
    assert is_dimensions_same(affinity_mat, propagated_mat)

    refined_matrix = np.zeros_like(affinity_mat)
    n = affinity_mat.shape[0]
    for i in range(n):
        for j in range(n):
            if propagated_mat[i, j] >= 0:
                refined_matrix[i, j] = 1 - (1 - propagated_mat[i, j]) * (1 - affinity_mat[i, j])
            else:
                refined_matrix[i, j] = (1 + propagated_mat[i, j]) * affinity_mat[i, j]
    return refined_matrix


class BasicPropagation(object):

    def propagate(self, affinity_mat, constraints_mat) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, affinity_mat, constraints_mat):
        propagated_mat = self.propagate(affinity_mat, constraints_mat)
        result_mat = affinity_matrix_refinement(affinity_mat, propagated_mat)
        return result_mat


class E2CPPropagation(BasicPropagation):
    """
        E2CP Propagation
    """

    def __init__(self, alpha, knn_k=0, temperature=1.0):
        self.alpha = alpha
        self.knn_k = knn_k
        self.temperature = temperature

    def do_knn(self, affinity_mat):
        if self.knn_k == -1:
            return affinity_mat
        n = affinity_mat.shape[0]
        if self.knn_k == 0:
            # default use this
            k = int(np.floor(np.log2(n)) + 1)
        else:
            k = self.knn_k
        knn_distances = np.sort(affinity_mat, axis=1)[:, 1:k + 1]
        knn_indices = np.argsort(affinity_mat, axis=1)[:, 1:k + 1]
        sigma = np.mean(knn_distances, axis=1)

        result_affinity_mat = np.zeros_like(affinity_mat)
        for i in range(n):
            for j in range(k):
                idx = knn_indices[i, j]
                distance = affinity_mat[i, idx]
                result_affinity_mat[i, idx] = np.exp(- (distance ** 2) / (self.temperature * sigma[i] * sigma[idx]))

        affinity_mat = (result_affinity_mat + result_affinity_mat.T) / 2.0
        return affinity_mat

    def compute_laplacian_with_knn(self, affinity_mat):
        affinity_mat = self.do_knn(affinity_mat)

        degree = np.diag(np.sum(affinity_mat, axis=1))
        # need to check formula
        degree_norm = np.diag(1.0 / (np.sqrt(np.diag(degree)) + 1e-10))
        laplacian_matrix = degree_norm.dot(affinity_mat).dot(degree_norm)

        return laplacian_matrix

    def propagate(self, affinity_mat, constraints_matrix):
        """
            Args:
                affinity_mat: np.ndarray
                constraints_matrix: np.ndarray
            Return:
                propagated_matrix: np.ndarray
        """
        embedding_num = affinity_mat.shape[0]

        laplacian_matrix = self.compute_laplacian_with_knn(affinity_mat)

        coefficient_matrix = np.linalg.inv(np.eye(embedding_num) - self.alpha * laplacian_matrix)
        propagated_matrix = coefficient_matrix.dot(constraints_matrix).dot(coefficient_matrix)
        propagated_matrix = (1 - self.alpha) ** 2 * propagated_matrix

        return propagated_matrix
