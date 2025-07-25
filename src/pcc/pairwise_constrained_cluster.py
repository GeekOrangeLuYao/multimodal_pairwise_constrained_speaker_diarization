import numpy as np

from utils.logger import get_logger

EPS = 1e-10

logger = get_logger()


def compute_laplacian(affinity, laplacian_type="GraphCut", eps=EPS):
    degree = np.diag(np.sum(affinity, axis=1))
    laplacian = degree - affinity
    if laplacian_type not in ['Unnormalized', 'RandomWalk', 'GraphCut']:
        raise TypeError("laplacian_type must be a LaplacianType")
    elif laplacian_type == 'Unnormalized':
        return laplacian
    elif laplacian_type == 'RandomWalk':
        # Random walk normalized version
        degree_norm = np.diag(1 / (np.diag(degree) + eps))
        laplacian_norm = degree_norm.dot(laplacian)
        return laplacian_norm
    elif laplacian_type == 'GraphCut':
        # Graph cut normalized version
        degree_norm = np.diag(1 / (np.sqrt(np.diag(degree)) + eps))
        laplacian_norm = degree_norm.dot(laplacian).dot(degree_norm)
        return laplacian_norm
    else:
        raise ValueError(f"Unsupported laplacian_type {laplacian_type}")


def compute_sorted_eigenvectors(input_matrix, descend=True):
    # Eigen decomposition.
    eigenvalues, eigenvectors = np.linalg.eig(input_matrix)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    if descend:
        # Sort from largest to smallest.
        index_array = np.argsort(-eigenvalues)
    else:
        # Sort from smallest to largest.
        index_array = np.argsort(eigenvalues)
    # Re-order.
    w = eigenvalues[index_array]
    v = eigenvectors[:, index_array]
    return w, v


def compute_number_of_clusters(eigenvalues,
                               max_clusters=None,
                               stop_eigenvalue=1e-2,
                               eigengap_type="ratio",
                               descend=True,
                               eps=EPS):
    if eigengap_type not in ['ratio', 'diff']:
        raise TypeError("eigengap_type must be a EigenGapType")
    max_delta = 0
    max_delta_index = 0
    range_end = len(eigenvalues)
    if max_clusters and max_clusters + 1 < range_end:
        range_end = max_clusters + 1

    eigenvalue_str = "eigenvalue gap:{\n"
    if not descend:
        # The first eigen value is always 0 in an ascending order
        for i in range(1, range_end - 1):
            if eigengap_type == "ratio":
                delta = eigenvalues[i + 1] / (eigenvalues[i] + eps)
                eigenvalue_str += f"\t{i} ({eigenvalues[i]:5f}) -> {i + 1} ({eigenvalues[i + 1]:5f}): {delta}\n"
            elif eigengap_type == "diff":
                delta = (eigenvalues[i + 1] - eigenvalues[i]) / np.max(eigenvalues)
                eigenvalue_str += f"\t{i} ({eigenvalues[i]:5f}) -> {i + 1} ({eigenvalues[i + 1]:5f}): {delta}\n"
            else:
                raise ValueError(f"Unsupported eigengap_type {eigengap_type}")
            if delta > max_delta:
                max_delta = delta
                max_delta_index = i + 1  # Index i means i+1 clusters
    else:
        for i in range(1, range_end):
            if eigenvalues[i - 1] < stop_eigenvalue:
                break
            if eigengap_type == "ratio":
                delta = eigenvalues[i - 1] / (eigenvalues[i] + eps)
                eigenvalue_str += f"\t{i - 1} ({eigenvalues[i - 1]:5f}) -> {i} ({eigenvalues[i]:5f}): {delta}\n"
            elif eigengap_type == "diff":
                delta = (eigenvalues[i - 1] - eigenvalues[i]) / np.max(eigenvalues)
                eigenvalue_str += f"\t{i - 1} ({eigenvalues[i - 1]:5f}) -> {i} ({eigenvalues[i]:5f}): {delta}\n"
            else:
                raise ValueError(f"Unsupported eigengap_type {eigengap_type}")
            if delta > max_delta:
                max_delta = delta
                max_delta_index = i
    eigenvalue_str += "}"
    # logger.info(f"\n{eigenvalue_str}")
    return max_delta_index, max_delta


class PairwiseConstrainedSpectralCluster(object):

    def __init__(self,
                 affinity_function,
                 refinements_list,
                 propagation_core,
                 laplacian_type,
                 eigengap_type,
                 post_process_cluster,
                 min_clusters=2,
                 max_clusters=10,
                 row_wise_re_norm=False,
                 custom_dist='cosine',
                 max_iter=300):
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters

        self.affinity_function = affinity_function
        self.refinements_list = refinements_list
        self.propagation_core = propagation_core
        self.laplacian_type = laplacian_type
        self.eigengap_type = eigengap_type
        self.post_process_cluster = post_process_cluster
        self.row_wise_re_norm = row_wise_re_norm
        self.custom_dist = custom_dist
        self.max_iter = max_iter

    def __call__(self, embedding_dict, constraints_dict):
        num_embeddings = len(embedding_dict)
        embedding_id_list = list(embedding_dict.keys())
        # build affinity_mat and constraints_mat
        affinity_mat, constraints_mat = self.affinity_function(
            embedding_dict, constraints_dict
        )

        # constraints propagation
        propagated_mat = self.propagation_core(affinity_mat, constraints_mat)

        # do refinement operations
        for refinement_operation in self.refinements_list:
            propagated_mat = refinement_operation(propagated_mat)

        laplacian_norm_mat = compute_laplacian(propagated_mat, laplacian_type=self.laplacian_type)

        eigenvalues, eigenvectors = compute_sorted_eigenvectors(laplacian_norm_mat, descend=False)
        n_clusters, max_delta_norm = compute_number_of_clusters(
            eigenvalues, max_clusters=self.max_clusters, eigengap_type=self.eigengap_type, descend=False
        )

        cut_n_clusters = n_clusters
        if self.min_clusters is not None:
            n_clusters = max(n_clusters, self.min_clusters)
            cut_n_clusters = n_clusters

        spectral_embeddings = eigenvectors[:, :cut_n_clusters]
        if self.row_wise_re_norm:
            rows_norm = np.linalg.norm(spectral_embeddings, axis=1, ord=2)
            spectral_embeddings = spectral_embeddings / np.reshape(rows_norm, (num_embeddings, 1))

        logger.info(f"Final get the speaker num is {n_clusters}")
        labels = self.post_process_cluster(
            spectral_embeddings=spectral_embeddings,
            n_clusters=n_clusters,
            custom_dist=self.custom_dist,
            max_iter=self.max_iter
        )

        embedding2cluster_dict = dict()
        for emb_id, label in zip(embedding_id_list, labels):
            embedding2cluster_dict[emb_id] = label

        return embedding2cluster_dict
