"""
PyTorch Implementation of SGL
Adapted from ID-GRec (https://github.com/BlueGhostYi/ID-GRec).
Self-supervised Graph Learning for Recommendation  Wu et al. SIGIR'21
"""

import torch
from tqdm import tqdm
from torch import nn
import utility.utility_function.losses as losses
import utility.utility_function.tools as tools
import utility.utility_data.data_graph
import utility.utility_train.batch_test as batch_test
import utility.utility_train.trainer as trainer
from time import time


class SGL(nn.Module):
    def __init__(self, config, dataset, device):
        super(SGL, self).__init__()
        self.config = config
        self.dataset = dataset
        self.device = device
        self.reg_lambda = float(self.config['reg_lambda'])
        self.ssl_lambda = float(self.config['ssl_lambda'])
        self.temperature = float(self.config['temperature'])
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_users,
                                                 embedding_dim=int(self.config['embedding_size']))
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_items,
                                                 embedding_dim=int(self.config['embedding_size']))

        # no pretrain
        nn.init.xavier_uniform_(self.user_embedding.weight, gain=1)
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=1)

        self.Graph = utility.utility_data.data_graph.sparse_adjacency_matrix(self.dataset)
        self.Graph = tools.convert_sp_mat_to_sp_tensor(self.Graph)
        self.Graph = self.Graph.coalesce().to(self.device)

        self.activation = nn.Sigmoid()

    def aggregate(self, graph):
        # [user + item, emb_dim]
        all_embedding = torch.cat([self.user_embedding.weight, self.item_embedding.weight])

        # no dropout
        embeddings = [all_embedding]

        for layer in range(int(self.config['GCN_layer'])):
            if isinstance(graph, list):
                all_embedding = torch.sparse.mm(graph[layer], all_embedding)
            else:
                all_embedding = torch.sparse.mm(graph, all_embedding)
            embeddings.append(all_embedding)

        final_embeddings = torch.stack(embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)

        users_emb, items_emb = torch.split(final_embeddings, [self.dataset.num_users, self.dataset.num_items])

        return users_emb, items_emb

    def forward(self, user, positive, negative, sub_graph_1, sub_graph_2):
        all_user_embeddings, all_item_embeddings = self.aggregate(self.Graph)

        user_embedding_1, item_embedding_1 = self.aggregate(sub_graph_1)
        user_embedding_2, item_embedding_2 = self.aggregate(sub_graph_2)

        user_embedding = all_user_embeddings[user.long()]
        pos_embedding = all_item_embeddings[positive.long()]
        neg_embedding = all_item_embeddings[negative.long()]

        ego_user_emb = self.user_embedding(user)
        ego_pos_emb = self.item_embedding(positive)
        ego_neg_emb = self.item_embedding(negative)

        bpr_loss = losses.get_bpr_loss(user_embedding, pos_embedding, neg_embedding)

        reg_loss = losses.get_reg_loss(ego_user_emb, ego_pos_emb, ego_neg_emb)
        reg_loss = self.reg_lambda * reg_loss

        user_index = user.long()
        item_index = positive.long()

        user_ssl_loss = losses.get_InfoNCE_loss(user_embedding_1[user_index], user_embedding_2[user_index], self.temperature)
        item_ssl_loss = losses.get_InfoNCE_loss(item_embedding_1[item_index], item_embedding_2[item_index], self.temperature)

        ssl_loss = self.ssl_lambda * (user_ssl_loss + item_ssl_loss)

        loss_list = [bpr_loss, reg_loss, ssl_loss]

        return loss_list

    def get_rating_for_test(self, user):
        all_user_embeddings, all_item_embeddings = self.aggregate(self.Graph)

        user_embeddings = all_user_embeddings[user.long()]

        rating = self.activation(torch.matmul(user_embeddings, all_item_embeddings.t()))
        return rating


class Trainer():
    def __init__(self, args, config, dataset, device, logger):
        self.model = SGL(config, dataset, device)
        self.args = args
        self.device = device
        self.config = config
        self.dataset = dataset
        self.logger = logger
        self.aug_type = config['aug_type']
        self.ssl_ratio = float(config['ssl_ratio'])

    # Customized training and testing process for SGL
    def train(self):
        self.SGL_trainer()

    def SGL_trainer(self):
        self.model.to(self.device)
        parameter_count = tools.format_parameter_count(self.model)
        self.logger.info("Model parameters: %s", parameter_count)

        Optim = torch.optim.Adam(self.model.parameters(), lr=float(self.config['learn_rate']))
        self.logger.info("Optimizer Initialized: Adam lr=%s", self.config['learn_rate'])

        loop_started_at = time()
        total_epochs = int(self.config['training_epochs'])
        validation_interval = int(self.config.get('interval', 1))
        if validation_interval <= 0:
            raise ValueError("interval must be a positive integer")
        self.logger.info("Validation interval: every %d epoch(s)", validation_interval)
        start_epoch, best_results = trainer.prepare_resumed_training(
            self.model, Optim, self.args, self.config, self.dataset,
            self.device, self.logger
        )
        for epoch in range(start_epoch, total_epochs):
            self.dataset.current_epoch = epoch + 1
            self.logger.info("Start Epoch: %d", epoch + 1)
            start_time = time()

            if self.aug_type in ['nd', 'ed']:
                sub_graph_1 = tools.create_adj_mat(self.dataset.user_item_net, self.aug_type, self.ssl_ratio)
                sub_graph_1 = tools.convert_sp_mat_to_sp_tensor(sub_graph_1).to(self.device)

                sub_graph_2 = tools.create_adj_mat(self.dataset.user_item_net, self.aug_type, self.ssl_ratio)
                sub_graph_2 = tools.convert_sp_mat_to_sp_tensor(sub_graph_2).to(self.device)
            else:
                sub_graph_1, sub_graph_2 = [], []
                for _ in range(0, int(self.config['GCN_layer'])):
                    temp_graph = tools.create_adj_mat(self.dataset.user_item_net, self.aug_type, self.ssl_ratio)
                    sub_graph_1.append(tools.convert_sp_mat_to_sp_tensor(temp_graph).to(self.device))

                    temp_graph = tools.create_adj_mat(self.dataset.user_item_net, self.aug_type, self.ssl_ratio)
                    sub_graph_2.append(tools.convert_sp_mat_to_sp_tensor(temp_graph).to(self.device))

            self.model.train()

            sample_data = self.dataset.sample_data_to_train_all()
            users = torch.Tensor(sample_data[:, 0]).long()
            pos_items = torch.Tensor(sample_data[:, 1]).long()
            neg_items = torch.Tensor(sample_data[:, 2]).long()

            users = users.to(self.device)
            pos_items = pos_items.to(self.device)
            neg_items = neg_items.to(self.device)

            users, pos_items, neg_items = tools.shuffle(users, pos_items, neg_items)
            num_batch = len(users) // int(self.config['batch_size']) + 1

            total_loss_list = []

            batches = tools.mini_batch(
                users,
                pos_items,
                neg_items,
                batch_size=int(self.config['batch_size']),
            )
            with tqdm(
                enumerate(batches),
                desc="Training epoch " + str(epoch + 1),
                total=num_batch,
            ) as progress:
                progress.set_postfix(params=parameter_count)
                for batch_i, (batch_users, batch_positive, batch_negative) in progress:
                    loss_list = self.model(batch_users, batch_positive, batch_negative, sub_graph_1, sub_graph_2)

                    if batch_i == 0:
                        assert len(loss_list) >= 1
                        total_loss_list = [0.] * len(loss_list)

                    total_loss = 0.
                    for i in range(len(loss_list)):
                        loss = loss_list[i]
                        total_loss += loss
                        total_loss_list[i] += loss.item()

                    Optim.zero_grad()
                    total_loss.backward()
                    Optim.step()

            end_time = time()

            loss_strs = str(round(sum(total_loss_list) / num_batch, 6)) \
                        + " = " + " + ".join([str(round(i / num_batch, 6)) for i in total_loss_list])

            self.logger.info("Train Loss: epoch=%d loss=%s", epoch + 1, loss_strs)
            self.logger.info("End Epoch: %d training_time=%.3f", epoch + 1, end_time - start_time)
            if tools.should_run_validation(epoch, total_epochs, validation_interval):
                _, best_results = batch_test.general_test(
                    self.dataset, self.model, self.device, self.config, epoch,
                    best_results, Optim, self.logger
                )
            elapsed_total = time() - loop_started_at
            self.logger.info(
                "Epoch Progress: %d/%d elapsed=%s ETA=%s",
                epoch + 1,
                total_epochs,
                tools.format_duration(elapsed_total),
                tools.estimate_remaining_time(elapsed_total, epoch + 1, total_epochs),
            )
            if best_results.get("stop", 0) > 0:
                break

        self.logger.info('Training loop completed.')
        self.logger.info("Best epoch: %4d | Best recall: %s | Best NDCG: %s" % (
        best_results['epoch'], best_results['recall'], best_results['ndcg']))
