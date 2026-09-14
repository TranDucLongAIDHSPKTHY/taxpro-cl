import numpy as np
import scipy.sparse as sp
import warnings
import logging
from config_path.config_path import adjacency_cache_file
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


def _build_bipartite_adjacency(data):
    """Build [[0, R], [R.T, 0]] without materializing a dense user-item block."""
    interaction_matrix = data.user_item_net.tocsr().astype(np.float32, copy=False)
    adjacency_matrix = sp.bmat(
        [[None, interaction_matrix], [interaction_matrix.T, None]],
        format="csr",
        dtype=np.float32,
    )
    expected_nonzero = 2 * interaction_matrix.nnz
    if adjacency_matrix.shape != (data.num_nodes, data.num_nodes):
        raise ValueError(
            "Unexpected adjacency shape: {} instead of {}".format(
                adjacency_matrix.shape, (data.num_nodes, data.num_nodes)
            )
        )
    if adjacency_matrix.nnz != expected_nonzero:
        raise ValueError(
            "Unexpected adjacency nonzero count: {} instead of {}".format(
                adjacency_matrix.nnz, expected_nonzero
            )
        )
    logger.info(
        "Sparse bipartite adjacency assembled: shape=%s nonzero=%d dtype=%s",
        adjacency_matrix.shape,
        adjacency_matrix.nnz,
        adjacency_matrix.dtype,
    )
    return adjacency_matrix


def sparse_adjacency_matrix_with_self(data):
    try:
        norm_adjacency = sp.load_npz(adjacency_cache_file(data.path, "with_self"))
        logger.info("Adjacency matrix loading completed: pre_A_with_self.npz")
    except Exception as error:
        if isinstance(error, FileNotFoundError):
            logger.info("Adjacency cache not found; building pre_A_with_self.npz")
        else:
            logger.exception("Unable to load pre_A_with_self.npz; rebuilding adjacency matrix")
        adjacency_matrix = _build_bipartite_adjacency(data)
        adjacency_matrix = adjacency_matrix + sp.eye(adjacency_matrix.shape[0])

        row_sum = np.array(adjacency_matrix.sum(axis=1))
        d_inv = np.power(row_sum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        degree_matrix = sp.diags(d_inv)

        norm_adjacency = degree_matrix.dot(adjacency_matrix).dot(degree_matrix).tocsr()
        sp.save_npz(adjacency_cache_file(data.path, "with_self", suffix=False), norm_adjacency)
        logger.info("Adjacency matrix constructed: pre_A_with_self.npz")

    return norm_adjacency


def sparse_adjacency_matrix(data):
    try:
        norm_adjacency = sp.load_npz(adjacency_cache_file(data.path, "adjacency"))
        logger.info("Adjacency matrix loading completed: pre_A.npz")
    except Exception as error:
        if isinstance(error, FileNotFoundError):
            logger.info("Adjacency cache not found; building pre_A.npz")
        else:
            logger.exception("Unable to load pre_A.npz; rebuilding adjacency matrix")
        adjacency_matrix = _build_bipartite_adjacency(data)

        row_sum = np.array(adjacency_matrix.sum(axis=1))
        d_inv = np.power(row_sum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        degree_matrix = sp.diags(d_inv)

        norm_adjacency = degree_matrix.dot(adjacency_matrix).dot(degree_matrix).tocsr()
        sp.save_npz(adjacency_cache_file(data.path, "adjacency", suffix=False), norm_adjacency)
        logger.info("Adjacency matrix constructed: pre_A.npz")

    return norm_adjacency


def sparse_adjacency_matrix_R(data):
    try:
        norm_adjacency = sp.load_npz(adjacency_cache_file(data.path, "interaction"))
        logger.info("Adjacency matrix loading completed: pre_R.npz")
    except Exception as error:
        if isinstance(error, FileNotFoundError):
            logger.info("Adjacency cache not found; building pre_R.npz")
        else:
            logger.exception("Unable to load pre_R.npz; rebuilding adjacency matrix")
        adjacency_matrix = data.user_item_net

        row_sum = np.array(adjacency_matrix.sum(axis=1))
        row_d_inv = np.power(row_sum, -0.5).flatten()
        row_d_inv[np.isinf(row_d_inv)] = 0.
        row_degree_matrix = sp.diags(row_d_inv)

        col_sum = np.array(adjacency_matrix.sum(axis=0))
        col_d_inv = np.power(col_sum, -0.5).flatten()
        col_d_inv[np.isinf(col_d_inv)] = 0.
        col_degree_matrix = sp.diags(col_d_inv)

        norm_adjacency = row_degree_matrix.dot(adjacency_matrix).dot(col_degree_matrix).tocsr()
        sp.save_npz(adjacency_cache_file(data.path, "interaction", suffix=False), norm_adjacency)
        logger.info("Adjacency matrix constructed: pre_R.npz")

    return norm_adjacency
