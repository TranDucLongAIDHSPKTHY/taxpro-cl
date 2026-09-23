import importlib
import json
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import scipy.sparse as sp
import torch

from tests.helpers import ToyDataset


ROOT = Path(__file__).resolve().parents[1]
if "faiss" not in sys.modules:
    fake_faiss = types.ModuleType("faiss")
    fake_faiss.Kmeans = object
    fake_faiss.get_num_gpus = lambda: 0
    sys.modules["faiss"] = fake_faiss


def config_for(_model):
    return {
        "embedding_size": "4",
        "reg_lambda": "0.0001",
        "learn_rate": "0.001",
        "GCN_layer": "2",
        "ssl_lambda": "0.1",
        # A high toy-only temperature avoids exp overflow on the tiny
        # identity graph used by this smoke test.
        "temperature": "10.0",
        "epsilon": "0.05",
        "cl_layer": "1",
        "alpha": "1.5",
        "k": "2",
        "proto_lambda": "1e-7",
        "svd_q": "2",
        "ssl_ratio": "0.1",
        "aug_type": "ed",
    }


class BaselineSmokeTests(unittest.TestCase):
    def test_six_baselines_import_one_batch_and_rating_smoke(self):
        dataset = ToyDataset()
        square = sp.eye(7, dtype=np.float32, format="csr")
        rectangular = dataset.user_item_net
        users = torch.tensor([0, 1])
        positive = torch.tensor([0, 2])
        negative = torch.tensor([3, 0])
        with mock.patch(
            "utility.utility_data.data_graph.sparse_adjacency_matrix",
            return_value=square,
        ), mock.patch(
            "utility.utility_data.data_graph.sparse_adjacency_matrix_R",
            return_value=rectangular,
        ):
            for name in (
                "LightGCN",
                "SimGCL",
                "XSimGCL",
                "SGL",
                "NCL",
            ):
                torch.manual_seed(42)
                np.random.seed(42)
                module = importlib.import_module("models." + name)
                model = getattr(module, name)(
                    config_for(name), dataset, torch.device("cpu")
                )
                if name == "SGL":
                    losses = model(users, positive, negative, model.Graph, model.Graph)
                elif name == "NCL":
                    losses = model(users, positive, negative, epoch=0)
                else:
                    losses = model(users, positive, negative)
                self.assertTrue(
                    all(torch.isfinite(value).all() for value in losses), name
                )
                rating = model.get_rating_for_test(users)
                self.assertEqual(tuple(rating.shape), (2, 4), name)
