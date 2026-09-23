from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np
import scipy.sparse as sp
import torch


class ToyDataset:
    def __init__(self):
        self.num_users = 3
        self.num_items = 4
        self.num_nodes = 7
        rows = np.asarray([0, 0, 1, 1, 2, 2])
        cols = np.asarray([0, 1, 1, 2, 2, 3])
        values = np.ones(len(rows), dtype=np.float32)
        self.inter_graph = sp.csr_matrix(
            (values, (rows, cols)), shape=(self.num_users, self.num_items)
        )
        self.user_item_net = self.inter_graph
        self.path = Path(".")
        # Same computation as the real Data.get_user_pos_items -- kept here
        # so ToyDataset satisfies the same interface for code that needs
        # each user's full train item list.
        self.all_positive = [
            self.user_item_net[user].nonzero()[1] for user in range(self.num_users)
        ]


class FakeTaxonomy:
    def __init__(self):
        self.dataset = "toy"
        self.policy = "no_merge"
        self.taxonomy_hash = "toy-taxonomy-hash"
        self.manifest = {"protocol_version": "taxprocl-p0-week3-v1"}
        self.statistics = {"catalog_items": 4}
        self.item_to_leaf_id = np.asarray([0, 0, 1, -1], dtype=np.int64)
        self.valid_taxonomy_mask = np.asarray([True, True, True, False])
        self.train_observed_mask = np.asarray([True, True, True, True])
        self.valid_train_mask = (
            self.valid_taxonomy_mask & self.train_observed_mask
        )
        self.num_items = 4
        self.num_prototypes = 2

    def as_torch(self, device=None):
        return {
            "item_to_leaf_id": torch.tensor(
                self.item_to_leaf_id, dtype=torch.long, device=device
            ),
            "valid_taxonomy_mask": torch.tensor(
                self.valid_taxonomy_mask, dtype=torch.bool, device=device
            ),
            "train_observed_mask": torch.tensor(
                self.train_observed_mask, dtype=torch.bool, device=device
            ),
            "valid_train_mask": torch.tensor(
                self.valid_train_mask, dtype=torch.bool, device=device
            ),
        }


def base_config():
    return {
        "dataset": "toy",
        "dataset_path": "./dataset_verify/",
        "top_K": "[10, 20]",
        "selection_K": "20",
        "training_epochs": "4",
        "early_stopping": "2",
        "interval": "1",
        "embedding_size": "4",
        "batch_size": "2",
        "test_batch_size": "2",
        "num_worker": "1",
        "learn_rate": "0.001",
        "reg_lambda": "0.0001",
        "GCN_layer": "1",
        "warm_start_epochs": "2",
        "mu": "0.9",
        "epsilon_max": "0.1",
        "delta": "1e-8",
        "temperature": "0.2",
        "ssl_lambda": "0.1",
        "taxonomy_policy": "no_merge",
        "taxonomy_granularity": "leaf",
        "prototype_mode": "leaf",
        "prototype_update": "ema",
        "mixture_alpha": "0.5",
        "augmentation_direction": "taxonomy",
        "unknown_taxonomy_policy": "exclude_cl",
        "symmetric_info_nce": "false",
        "resume_checkpoint": "none",
        "sparsity_test": "0",
    }


@contextmanager
def patched_taxpro_dependencies():
    # models/TaxPro-CL.py is self-contained (see its class docstring), so only its own module
    # namespace needs patching. evaluation_protocol_dir/read_json are left
    # to callers (e.g. tests/test_taxpro_cl_improved.py's
    # patched_improved_dependencies), which need call-order-specific
    # side_effect values, not a single shared return_value here.
    fake = FakeTaxonomy()
    with mock.patch(
        "models.TaxPro-CL.load_taxonomy", return_value=fake
    ), mock.patch(
        "models.TaxPro-CL.sha256_file", return_value="toy-evaluation-hash"
    ), mock.patch(
        "models.TaxPro-CL.utility.utility_data.data_graph.sparse_adjacency_matrix",
        return_value=sp.eye(7, dtype=np.float32, format="csr"),
    ):
        yield fake


def make_fake_parent_taxonomy():
    """A parent taxonomy where all 3 valid ToyDataset items collapse into a
    single bucket -- deliberately different from FakeTaxonomy's 2-leaf split,
    so mixture-mode tests can tell leaf and parent directions apart."""
    parent = FakeTaxonomy()
    parent.item_to_leaf_id = [0, 0, 0, -1]
    parent.valid_taxonomy_mask = [True, True, True, False]
    parent.num_prototypes = 1
    return parent


@contextmanager
def patched_taxpro_dependencies_mixture():
    leaf_taxonomy = FakeTaxonomy()
    parent_taxonomy = make_fake_parent_taxonomy()
    with mock.patch(
        "models.TaxPro-CL.load_taxonomy",
        side_effect=[leaf_taxonomy, parent_taxonomy],
    ), mock.patch(
        "models.TaxPro-CL.sha256_file", return_value="toy-evaluation-hash"
    ), mock.patch(
        "models.TaxPro-CL.utility.utility_data.data_graph.sparse_adjacency_matrix",
        return_value=sp.eye(7, dtype=np.float32, format="csr"),
    ):
        yield leaf_taxonomy, parent_taxonomy
