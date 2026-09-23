import os
import torch
import random
import numpy as np
import scipy.sparse as sp
import logging

logger = logging.getLogger(__name__)


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def format_duration(seconds):
    """Format a duration as HH:MM:SS for stable console and file logs."""
    total_seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)


def count_parameters(model, trainable_only=False):
    """Return the number of parameters in a PyTorch model."""
    parameters = model.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def format_parameter_count(model, trainable_only=False):
    """Format a model parameter count for logs and progress bars."""
    return "{:,}".format(count_parameters(model, trainable_only=trainable_only))


def estimate_remaining_time(elapsed_seconds, completed_steps, total_steps):
    """Estimate remaining time from the average duration of completed steps."""
    completed_steps = int(completed_steps)
    total_steps = int(total_steps)
    if completed_steps <= 0:
        return "--:--:--"
    remaining_steps = max(total_steps - completed_steps, 0)
    average_seconds = float(elapsed_seconds) / completed_steps
    return format_duration(average_seconds * remaining_steps)


def should_run_validation(epoch, total_epochs, interval):
    """Run validation at each interval boundary and always at the final epoch."""
    interval = int(interval)
    if interval <= 0:
        raise ValueError("interval must be a positive integer")
    completed_epoch = int(epoch) + 1
    return completed_epoch % interval == 0 or completed_epoch == int(total_epochs)


def read_configuration(filename, model):
    if not os.path.exists(filename):
        logger.error("The path does not have a configuration file for %s: %s", model, filename)
        raise IOError("Configuration file not found: {}".format(filename))
    else:
        with open(filename, "r") as f:
            config = dict()
            line = f.readline()
            while line is not None and line != "":
                try:
                    name, value = line.strip().split("=")
                    config[name.strip()] = value.strip()
                except ValueError:
                    logger.exception("Configuration file format error: %s", filename)
                    raise
                line = f.readline()
        return config


def shuffle(*arrays, **kwargs):
    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('Inputs to shuffle must have the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


def mini_batch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size', 1024)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def create_adj_mat(inter_graph, aug_type, ssl_rate):
    graph_shape = inter_graph.get_shape()
    node_number = graph_shape[0] + graph_shape[1]
    user_index, item_index = inter_graph.nonzero()

    if aug_type == 'nd':
        raise NotImplementedError("The method does not implemented.")
    elif aug_type in ['ed', 'rw']:
        edge_number = inter_graph.count_nonzero()

        keep_index = random.sample(range(edge_number), k=int((1 - ssl_rate) * edge_number))
        user_index = np.array(user_index)[keep_index]
        item_index = np.array(item_index)[keep_index]
        ratings = np.ones_like(user_index, dtype=np.float32)
        new_graph = sp.csr_matrix((ratings, (user_index, item_index + graph_shape[0])), shape=(node_number, node_number))

    adjacency_matrix = new_graph + new_graph.T

    row_sum = np.array(adjacency_matrix.sum(axis=1))
    d_inv = np.power(row_sum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    degree_matrix = sp.diags(d_inv)

    norm_adjacency = degree_matrix.dot(adjacency_matrix).dot(degree_matrix).tocsr()

    return norm_adjacency


def convert_sp_mat_to_sp_tensor(sp_mat):
    """
        coo.row: x in user-item graph
        coo.col: y in user-item graph
        coo.data: [value(x,y)]
    """
    coo = sp_mat.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()

    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    value = torch.FloatTensor(coo.data)
    # from a sparse matrix to a sparse float tensor
    sp_tensor = torch.sparse_coo_tensor(index, value, torch.Size(coo.shape))
    return sp_tensor
