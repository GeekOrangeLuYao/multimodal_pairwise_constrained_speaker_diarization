"""
    Define several refinements
"""
import numpy as np
from scipy.ndimage import gaussian_filter

from utils.logger import get_logger

logger = get_logger()


def check_affinity_matrix(affinity):
    """
        Check the input to the refinement method.

        Args:
            affinity: the input affinity matrix
        Raises:
            TypeError: if affinity has wrong type
            ValueError: if affinity has wrong shape, etc.
    """
    if not isinstance(affinity, np.ndarray):
        raise TypeError("affinity must be a numpy array")
    shape = affinity.shape
    if len(shape) != 2:
        raise ValueError("affinity must be 2-dimensional")
    if shape[0] != shape[1]:
        raise ValueError("affinity must be a square matrix")


class GaussianBlur(object):

    def __init__(self, sigma=1):
        self.sigma = sigma

    def __call__(self, affinity):
        check_affinity_matrix(affinity)
        return gaussian_filter(affinity, sigma=self.sigma)


class CropDiagonal(object):

    def __init__(self):
        pass

    def __call__(self, affinity):
        check_affinity_matrix(affinity)
        refined_affinity = np.copy(affinity)
        np.fill_diagonal(refined_affinity, 0.0)
        di = np.diag_indices(refined_affinity.shape[0])
        refined_affinity[di] = refined_affinity.max(axis=1)
        return refined_affinity


class RowWiseThreshold(object):
    """Apply row wise thresholding."""

    def __init__(self,
                 p_percentile=0.95,
                 thresholding_soft_multiplier=0.01,
                 thresholding_type="Percentile",
                 thresholding_with_binarization=False,
                 thresholding_preserve_diagonal=False):
        if thresholding_type not in ['RowMax', 'Percentile']:
            raise ValueError(f"Unknown thresholding_type {thresholding_type}")
        self.p_percentile = p_percentile
        self.multiplier = thresholding_soft_multiplier
        self.thresholding_type = thresholding_type
        self.thresholding_with_binarization = thresholding_with_binarization
        self.thresholding_preserve_diagonal = thresholding_preserve_diagonal

    def __call__(self, affinity):
        check_affinity_matrix(affinity)
        refined_affinity = np.copy(affinity)
        if self.thresholding_preserve_diagonal:
            np.fill_diagonal(refined_affinity, 0.0)
        if self.thresholding_type == "RowMax":
            # Row_max based thresholding
            row_max = refined_affinity.max(axis=1)
            row_max = np.expand_dims(row_max, axis=1)
            is_smaller = refined_affinity < (row_max * self.p_percentile)
        elif self.thresholding_type == "Percentile":
            # Percentile based thresholding
            row_percentile = np.percentile(
                refined_affinity, self.p_percentile * 100, axis=1
            )
            row_percentile = np.expand_dims(row_percentile, axis=1)
            is_smaller = refined_affinity < row_percentile
        else:
            raise ValueError(f"Unsupported thresholding_type {self.thresholding_type}")
        if self.thresholding_with_binarization:
            # For values larger than the threshold, we binarize them to 1
            refined_affinity = (np.ones_like(refined_affinity) * np.invert(is_smaller)) + \
                               (refined_affinity * self.multiplier * is_smaller)
        else:
            refined_affinity = (refined_affinity * np.invert(is_smaller)) + \
                               (refined_affinity * self.multiplier * is_smaller)
        if self.thresholding_preserve_diagonal:
            np.fill_diagonal(refined_affinity, 1.0)
        return refined_affinity


class RowWiseThreshold2(object):

    def __init__(self,
                 p_percentile,
                 min_p_num=6.0,
                 thresholding_soft_multiplier=0.0,
                 ):
        self.p_percentile = p_percentile
        self.min_p_num = min_p_num
        self.multiplier = thresholding_soft_multiplier

    def __call__(self, affinity):
        check_affinity_matrix(affinity)
        embedding_num = affinity.shape[0]

        p_percentile = self.p_percentile
        if embedding_num * (1 - self.p_percentile) < self.min_p_num:
            p_percentile = 1 - self.min_p_num / embedding_num

        refined_affinity = np.copy(affinity)

        row_percentile = np.percentile(
            refined_affinity, p_percentile * 100, axis=1
        )
        row_percentile = np.expand_dims(row_percentile, axis=1)
        is_smaller = refined_affinity < row_percentile

        refined_affinity = (refined_affinity * np.invert(is_smaller)) + \
                           (refined_affinity * self.multiplier * is_smaller)
        return refined_affinity


class Symmetrize(object):

    def __init__(self, symmetrize_type="average"):
        if symmetrize_type not in ['max', 'average']:
            raise ValueError(
                f"Symmetrize_type {symmetrize_type} not true should in ['max', 'average']"
            )
        self.symmetrize_type = symmetrize_type

    def __call__(self, affinity):
        check_affinity_matrix(affinity)
        if self.symmetrize_type == "max":
            return np.maximum(affinity, np.transpose(affinity))
        elif self.symmetrize_type == "average":
            return 0.5 * (affinity + np.transpose(affinity))
        else:
            raise ValueError(f"Unsupported symmetrize_type = {self.symmetrize_type}")


class Diffuse(object):

    def __init__(self):
        pass

    def __call__(self, affinity):
        check_affinity_matrix(affinity)
        return np.matmul(affinity, np.transpose(affinity))


class RowWiseNormalize(object):
    """The row wise max normalization operation."""

    def __init__(self) -> None:
        pass

    def __call__(self, affinity):
        check_affinity_matrix(affinity)
        refined_affinity = np.copy(affinity)
        row_max = refined_affinity.max(axis=1)
        refined_affinity /= np.expand_dims(row_max, axis=1)
        return refined_affinity


class RefinementList(object):

    refinement_dict = {
        "GaussianBlur": GaussianBlur,
        "CropDiagonal": CropDiagonal,
        "RowWiseThreshold": RowWiseThreshold,
        "RowWiseThreshold2": RowWiseThreshold2,
        "Symmetrize": Symmetrize,
        "Diffuse": Diffuse,
        "RowWiseNormalize": RowWiseNormalize
    }

    def __init__(self, refinements) -> None:
        logger.info(f"Build RefinementList: {refinements}")
        self.refinement_operations = []
        refinement_name_list = [
            refinement_name for refinement_name in refinements
        ]
        for refinement_name in self.refinement_dict:
            if refinement_name not in refinement_name_list:
                continue
            else:
                self.refinement_operations.append(
                    refinements[refinement_name]
                )
        assert len(self.refinement_operations) == len(refinement_name_list)

    def __iter__(self):
        for refinement_operation in self.refinement_operations:
            yield refinement_operation

