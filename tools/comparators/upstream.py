"""Runs a comparator from its authors' code, unmodified, and exports its final tables.

Executed by tools/experiments/run_comparators.py in the comparator environment, with the
authors' repositories checked out under the comparator directory at the pinned commits of
docs/extended_evaluation_protocol.md. Nothing here edits those repositories; every
adaptation is applied from this file at run time:

- KGCL: the settings of its Amazon dataset apply, so the prepared data is served under
  that name. Packages imported but not used on the executed path are replaced by empty
  stand-ins, `degree` by its definition, and data loaders run in this process.
- GALORE: the number of candidates of edge addition reaches `graph_add_edge`, whose
  default its command line does not change. Checkpoints are selected on validation
  Overall Recall@20 instead of the authors' four-metric vote at rank 50.

The authors' test slots receive validation data. The test split is never passed here.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import pickle
import random
import runpy
import sys
import types
from pathlib import Path

import numpy as np
import torch

TOP = 20
EVALUATION_BATCH = 1024
GALORE_DATASET = "office-products"
EMBEDDING_SIZE = 64
KGCL_LAYERS = 3


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def install_stand_ins():
    """KGCL imports these at module level; the executed path uses only `degree`."""
    utils = types.ModuleType("torch_geometric.utils")
    utils.degree = lambda index: torch.bincount(index).float()
    utils.to_undirected = None
    geometric = types.ModuleType("torch_geometric")
    geometric.utils = utils
    sparse_tensor = types.ModuleType("torch_sparse.tensor")
    sparse_tensor.to = None
    sparse = types.ModuleType("torch_sparse")
    sparse.tensor = sparse_tensor
    cpp = types.ModuleType("cppimport")
    cpp.imp = None
    board = types.ModuleType("tensorboardX")
    board.SummaryWriter = None
    sys.modules.update({"torch_geometric": geometric, "torch_geometric.utils": utils, "torch_sparse": sparse,
                        "torch_sparse.tensor": sparse_tensor, "cppimport": cpp, "tensorboardX": board})


def run_in_process_loaders():
    import torch.utils.data.dataloader as loader_module

    original = loader_module.DataLoader

    def single_process(*args, **kwargs):
        kwargs["num_workers"] = 0
        return original(*args, **kwargs)

    loader_module.DataLoader = single_process


def run_kgcl(args):
    code = Path(args.comparators) / "KGCL" / "code"
    os.chdir(code)
    sys.path.insert(0, str(code))
    install_stand_ins()
    sys.argv = ["main.py", "--dataset", "amazon-book", "--model", "kgcl", "--seed", str(args.seed),
                "--recdim", str(EMBEDDING_SIZE), "--layer", str(KGCL_LAYERS), "--bpr_batch", "2048",
                "--epochs", str(args.max_epochs)]
    import world

    world.DATA_PATH = str(Path(args.prepared) / "kgcl")
    world.FILE_PATH = str(args.output)
    world.kgc_temp = args.temperature
    world.ssl_reg = args.ssl_weight
    run_in_process_loaders()
    state = runpy.run_path("main.py", run_name="__main__")
    model = state["Recmodel"]
    model.load_state_dict(torch.load(state["weight_file"], map_location=world.device))
    model.eval()
    with torch.no_grad():
        users, items = model.computer()
    Path(state["weight_file"]).unlink()
    return users.cpu().numpy(), items.cpu().numpy(), int(state["epoch"]) - int(state["stopping_step"])


def validation_lists(rec, prepared):
    """Validation targets of each trained user, on the internal ids of the comparator."""
    targets = json.loads((Path(prepared) / "validation_overall.json").read_text(encoding="utf-8"))
    users, positives = [], []
    for user, items in targets.items():
        if user in rec.data.user:
            users.append(rec.data.user[user])
            positives.append(({rec.data.item[str(i)] for i in items if str(i) in rec.data.item}, len(items)))
    return users, positives


def validation_recall(user_table, item_table, seen, users, positives):
    total = 0.0
    for start in range(0, len(users), EVALUATION_BATCH):
        batch = users[start:start + EVALUATION_BATCH]
        scores = user_table[batch] @ item_table.transpose(0, 1)
        rows, columns = seen[batch].nonzero()
        scores[torch.as_tensor(rows, device=scores.device), torch.as_tensor(columns, device=scores.device)] = -np.inf
        top = torch.topk(scores, k=TOP).indices.cpu().tolist()
        for offset, ranked in enumerate(top):
            relevant, size = positives[start + offset]
            total += len(relevant.intersection(ranked)) / size
    return total / len(users)


def run_galore(args):
    code = Path(args.comparators) / "GALORE"
    os.chdir(code)
    sys.path.insert(0, str(code))
    from data.loader import FileIO
    from model.graph.GALORE import GALORE
    from util.conf import ModelConf

    conf = ModelConf("./conf/GALORE.conf")
    folder = f"./dataset/{GALORE_DATASET}"
    values = {"train_mode": "train_normal", "dataset": GALORE_DATASET, "embbedding.size": str(EMBEDDING_SIZE),
              "drop_rate": str(args.drop_rate), "add_edge": str(args.edge_num), "cluster_num": "1",
              "training.set": f"{folder}/train.txt", "valid.set": f"{folder}/valid.txt",
              "test.set": f"{folder}/test.txt", "test1.set": f"{folder}/test1.txt",
              "test2.set": f"{folder}/test2.txt"}
    for key, value in values.items():
        conf.__setitem__(key, value)
    sets = [FileIO.load_data_set(conf[name], "graph")
            for name in ("training.set", "test.set", "test1.set", "test2.set", "valid.set")]

    class SelectedOnValidation(GALORE):
        best_recall = -1.0
        best_epoch = None

        def fast_evaluation(self, epoch):
            recall = validation_recall(self.user_emb, self.item_emb, self.data.interaction_mat, self.users, self.positives)
            print(f"epoch {epoch + 1} validation Recall@{TOP} {recall:.6f}", flush=True)
            if recall > self.best_recall:
                self.best_recall, self.best_epoch = recall, epoch + 1
                self.best_user_emb, self.best_item_emb = self.user_emb.clone(), self.item_emb.clone()

    rec = SelectedOnValidation(conf, *sets)
    rec.users, rec.positives = validation_lists(rec, args.prepared)
    rec.model.graph_add_edge = functools.partial(rec.model.graph_add_edge, edge_num=args.edge_num)
    pretrained = np.load(args.pretrained)
    user_order = [int(rec.data.id2user[index]) for index in range(rec.data.user_num)]
    item_order = [int(rec.data.id2item[index]) for index in range(rec.data.item_num)]
    target = Path(f"./model_cpt/{GALORE_DATASET}")
    target.mkdir(parents=True, exist_ok=True)
    torch.save({"embedding_dict.user_emb": torch.as_tensor(pretrained["users"][user_order]),
                "embedding_dict.item_emb": torch.as_tensor(pretrained["items"][item_order])},
               target / f"lightgcn_best_{EMBEDDING_SIZE}.pt")
    rec.train()
    users = np.zeros((max(user_order) + 1, EMBEDDING_SIZE), dtype=np.float32)
    items = np.zeros((max(item_order) + 1, EMBEDDING_SIZE), dtype=np.float32)
    users[user_order] = rec.user_emb.detach().cpu().numpy()
    items[item_order] = rec.item_emb.detach().cpu().numpy()
    return users, items, rec.best_epoch


def prepare_galore(args):
    """The authors' popularity split and edge-weight dictionaries, from their own script.
    The popular list is stored as a set: the sampler tests membership for every pair."""
    directory = Path(args.comparators) / "GALORE" / "dataset" / GALORE_DATASET
    sys.path.insert(0, str(directory.parent))
    os.chdir(directory)
    import data_process

    data_process.testset_split()
    data_process.get_sim_dict()
    path = directory / "popular_item_list.pkl"
    with path.open("rb") as stream:
        popular = pickle.load(stream)
    with path.open("wb") as stream:
        pickle.dump(set(popular), stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=("kgcl", "galore", "prepare-galore"), required=True)
    parser.add_argument("--comparators", required=True)
    parser.add_argument("--prepared")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pretrained")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--ssl-weight", type=float)
    parser.add_argument("--edge-num", type=int)
    parser.add_argument("--drop-rate", type=float)
    parser.add_argument("--max-epochs", type=int, default=1000)
    args = parser.parse_args()
    if args.method == "prepare-galore":
        prepare_galore(args)
        return 0
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.method == "galore":
        args.pretrained = str(Path(args.pretrained).resolve())
        args.prepared = str(Path(args.prepared).resolve())
        set_seeds(args.seed)
        users, items, epoch = run_galore(args)
    else:
        args.prepared = str(Path(args.prepared).resolve())
        users, items, epoch = run_kgcl(args)
    np.savez(args.output / "embeddings.npz", users=users, items=items, best_epoch=epoch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
