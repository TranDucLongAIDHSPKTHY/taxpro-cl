import logging
import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.sparse as sp
from config_path.config_path import dataset_split_file

warnings.filterwarnings("ignore")


def _sample_negative_items(users, num_items, user_item_net, seed):
    """Vectorized negative sampling for one worker-owned user chunk."""
    if len(users) == 0:
        return np.empty(0, dtype=np.int64)
    positive_counts = np.diff(user_item_net.indptr)[users]
    if np.any(positive_counts >= num_items):
        raise ValueError("Cannot sample a negative item for a user connected to every item")

    random_state = np.random.RandomState(seed)
    negatives = random_state.randint(0, num_items, size=len(users)).astype(np.int64)
    positive_mask = np.asarray(user_item_net[users, negatives]).reshape(-1) > 0
    while np.any(positive_mask):
        negatives[positive_mask] = random_state.randint(
            0, num_items, size=int(positive_mask.sum())
        )
        positive_mask = np.asarray(user_item_net[users, negatives]).reshape(-1) > 0
    return negatives


class Data(object):
    def __init__(self, path, config, logger=None):
        self.path = path
        self.logger = logger or logging.getLogger(__name__)
        self.num_users = 0
        self.num_items = 0
        self.num_entities = 0
        self.num_relations = 0
        self.num_nodes = 0
        self.num_train = 0
        self.num_validation = 0
        self.num_test = 0
        self.current_epoch = "not started"
        self.training_output_dir = None
        self.validation_history = []
        self.config = config
        self.num_worker = max(1, int(config.get("num_worker", 2)))
        self.logger.info("Negative sampling workers: %d", self.num_worker)

        self.load_data()
        if config:
            self.split_test_dict = None
            self.split_state = None
            if "sparsity_test" in config and int(config["sparsity_test"]) == 1:
                self.split_test_dict, self.split_state = self.create_sparsity_split()

    def load_data(self):
        train_path = dataset_split_file(self.path, "train")
        validation_path = dataset_split_file(self.path, "validation")
        test_path = dataset_split_file(self.path, "test")
        for file_path in (train_path, validation_path, test_path):
            if not file_path.is_file():
                raise FileNotFoundError("Required dataset file not found: {}".format(file_path))

        _, self.train_user, self.train_item, self.num_train, self.pos_length = self.read_ratings(train_path)
        _, self.validation_user, self.validation_item, self.num_validation, _ = self.read_ratings(validation_path)
        _, self.test_user, self.test_item, self.num_test, _ = self.read_ratings(test_path)

        self.num_users += 1
        self.num_items += 1
        self.num_nodes = self.num_users + self.num_items
        assert len(self.train_user) == len(self.train_item)

        self.user_item_net = sp.csr_matrix(
            (np.ones(len(self.train_user)), (self.train_user, self.train_item)),
            shape=(self.num_users, self.num_items),
        )
        self.validation_user_item_net = sp.csr_matrix(
            (np.ones(len(self.validation_user)), (self.validation_user, self.validation_item)),
            shape=(self.num_users, self.num_items),
        )
        self.train_validation_net = (self.user_item_net + self.validation_user_item_net).tocsr()

        self.all_positive = self.get_user_pos_items(list(range(self.num_users)))
        self.validation_dict = self.build_interaction_dict(self.validation_user, self.validation_item)
        self.test_dict = self.build_interaction_dict(self.test_user, self.test_item)
        self.data_statistics()

    def read_ratings(self, file_name):
        inter_users, inter_items, unique_users = [], [], []
        inter_num = 0
        pos_length = []
        with open(file_name, "r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                temp = line.strip()
                if not temp:
                    continue
                try:
                    arr = [int(value) for value in temp.split()]
                except ValueError as error:
                    raise ValueError("Invalid rating at {}:{}".format(file_name, line_number)) from error
                user_id, pos_id = arr[0], arr[1:]
                unique_users.append(user_id)
                if not pos_id:
                    continue
                self.num_users = max(self.num_users, user_id)
                self.num_items = max(self.num_items, max(pos_id))
                inter_users.extend([user_id] * len(pos_id))
                pos_length.append(len(pos_id))
                inter_items.extend(pos_id)
                inter_num += len(pos_id)
        return np.array(unique_users), np.array(inter_users), np.array(inter_items), inter_num, pos_length

    def data_statistics(self):
        sparsity = 1 - (
            self.num_train + self.num_validation + self.num_test
        ) / self.num_users / self.num_items
        self.logger.info(
            "Dataset statistics: users=%d items=%d nodes=%d train=%d validation=%d test=%d sparsity=%.6f",
            self.num_users, self.num_items, self.num_nodes, self.num_train,
            self.num_validation, self.num_test, sparsity,
        )

    def get_statistics(self):
        sparsity = 1 - (
            self.num_train + self.num_validation + self.num_test
        ) / self.num_users / self.num_items
        return (
            "dataset:{}\tnum_users:{}, num_items:{}\t|num_train:{}, "
            "num_validation:{}, num_test:{}, sparsity:{:.6f}"
        ).format(
            self.config["dataset"], self.num_users, self.num_items, self.num_train,
            self.num_validation, self.num_test, sparsity,
        )

    def sample_data_to_train_random(self):
        users = np.random.randint(0, self.num_users, len(self.train_user))
        sample_list = []
        for user in users:
            positive_items = self.all_positive[user]
            if len(positive_items) == 0:
                continue
            positive_item = positive_items[np.random.randint(0, len(positive_items))]
            while True:
                negative_item = np.random.randint(0, self.num_items)
                if negative_item not in positive_items:
                    break
            sample_list.append([user, positive_item, negative_item])
        return np.array(sample_list)

    def sample_data_to_train_all(self):
        num_samples = len(self.train_user)
        num_workers = min(self.num_worker, max(num_samples, 1))
        index_chunks = np.array_split(np.arange(num_samples), num_workers)
        worker_seeds = [int(np.random.randint(0, 2 ** 31 - 1)) for _ in index_chunks]

        if num_workers == 1:
            negative_chunks = [
                _sample_negative_items(
                    self.train_user[index_chunks[0]],
                    self.num_items,
                    self.user_item_net,
                    worker_seeds[0],
                )
            ]
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(
                        _sample_negative_items,
                        self.train_user[indices],
                        self.num_items,
                        self.user_item_net,
                        worker_seed,
                    )
                    for indices, worker_seed in zip(index_chunks, worker_seeds)
                ]
                negative_chunks = [future.result() for future in futures]

        negative_items = np.concatenate(negative_chunks)
        return np.column_stack((self.train_user, self.train_item, negative_items))

    def get_user_pos_items(self, users):
        return [self.user_item_net[user].nonzero()[1] for user in users]

    def get_user_pos_items_for_evaluation(self, users, split):
        matrix = self.user_item_net if split == "validation" else self.train_validation_net
        return [matrix[user].nonzero()[1] for user in users]

    def get_user_n_neg_items(self, users, n):
        negative_items = []
        for user in users:
            negative_list = []
            for _ in range(n):
                while True:
                    negative_item = np.random.randint(0, self.num_items)
                    if negative_item not in self.all_positive[user]:
                        negative_list.append(negative_item)
                        break
            negative_items.append(negative_list)
        return negative_items

    @staticmethod
    def build_interaction_dict(users, items):
        interactions = {}
        for user, item in zip(users, items):
            interactions.setdefault(user, []).append(item)
        return interactions

    def build_test(self):
        return self.build_interaction_dict(self.test_user, self.test_item)

    def create_sparsity_split(self):
        all_users = list(self.test_dict.keys())
        user_n_iid = {}
        for uid in all_users:
            num_iids = len(self.train_validation_net[uid].nonzero()[1]) + len(self.test_dict[uid])
            user_n_iid.setdefault(num_iids, []).append(uid)

        split_uids, temp, split_state = [], [], []
        count, n_rates = 1, 0
        total_rates = self.num_train + self.num_validation + self.num_test
        n_count = total_rates
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            rates = n_iids * len(user_n_iid[n_iids])
            n_rates += rates
            n_count -= rates
            if n_rates >= count * 0.25 * total_rates:
                split_uids.append(temp)
                state = "#inter per user<=[{}], #users=[{}], #all rates=[{}]".format(
                    n_iids, len(temp), n_rates
                )
                split_state.append(state)
                self.logger.info("Sparsity split: %s", state)
                temp, n_rates = [], 0
                count += 1
            if idx == len(user_n_iid) - 1 or n_count == 0:
                split_uids.append(temp)
                state = "#inter per user<=[{}], #users=[{}], #all rates=[{}]".format(
                    n_iids, len(temp), n_rates
                )
                split_state.append(state)
                self.logger.info("Sparsity split: %s", state)
        return split_uids, split_state
