import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.utils as utils


class Policy_step(nn.Module):
    def __init__(self, m, embedding_size, hidden_size):
        super(Policy_step, self).__init__()

        self.batch_norm = nn.BatchNorm1d(m * hidden_size)
        # self.lstm_cell = nn.LSTMCell(input_size=2 * m * embedding_size, hidden_size=2 * m * hidden_size)
        self.lstm_cell = nn.LSTMCell(input_size=m * embedding_size, hidden_size=m * hidden_size)



    def forward(self, prev_action, prev_state):


        output, new_state = self.lstm_cell(prev_action, prev_state)


        return output, (output, new_state)

class Policy_mlp(nn.Module):
    def __init__(self, hidden_size, m, embedding_size):
        super(Policy_mlp, self).__init__()

        self.hidden_size = hidden_size
        self.m = m
        self.embedding_size = embedding_size
        # input: output + prev_entity + query_relation
        self.mlp_l1 = nn.Linear(2 * m * self.hidden_size, m * self.hidden_size, bias=True)
        self.mlp_l2 = nn.Linear(m * self.hidden_size, m * self.embedding_size, bias=True)

    def forward(self, state_query):
        hidden = torch.relu(self.mlp_l1(state_query))
        output = torch.relu(self.mlp_l2(hidden))
        return output

class Agent(nn.Module):

    def __init__(self, params):
        super(Agent, self).__init__()
        self.action_vocab_size = len(params['relation_vocab'])
        self.entity_vocab_size = len(params['entity_vocab'])
        self.relation_vocab_size = len(params['relation_vocab'])

        self.embedding_size = params['embedding_size']
        self.hidden_size = params['hidden_size']

        self.ePAD = params['entity_vocab']['PAD']
        self.rPAD = params['relation_vocab']['PAD']

        self.use_entity_embeddings = params['use_entity_embeddings']
        self.train_entity_embeddings = params['train_entity_embeddings']
        self.train_relation_embeddings = params['train_relation_embeddings']

        self.device = params['device']

        if self.use_entity_embeddings:
            if self.train_entity_embeddings:
                self.entity_embedding = nn.Embedding(self.entity_vocab_size, 2 * self.embedding_size)
            else:
                self.entity_embedding = nn.Embedding(self.entity_vocab_size, 2 * self.embedding_size).requires_grad_(
                    False)
            torch.nn.init.xavier_uniform_(self.entity_embedding.weight)
        else:
            if self.train_entity_embeddings:
                self.entity_embedding = nn.Embedding(self.entity_vocab_size, 2 * self.embedding_size)
            else:
                self.entity_embedding = nn.Embedding(self.entity_vocab_size, 2 * self.embedding_size).requires_grad_(
                    False)
            torch.nn.init.constant_(self.entity_embedding.weight, 0.0)

        if self.train_relation_embeddings:
            self.relation_embedding = nn.Embedding(self.relation_vocab_size, 2 * self.embedding_size)
        else:
            self.relation_embedding = nn.Embedding(self.relation_vocab_size, 2 * self.embedding_size).requires_grad_(
                False)
        torch.nn.init.xavier_uniform_(self.relation_embedding.weight)


        self.num_rollouts = params['num_rollouts']
        self.test_rollouts = params['test_rollouts']
        self.LSTM_Layers = params['LSTM_layers']
        self.batch_size = params['batch_size'] * params['num_rollouts']
        self.dummy_start_label = (torch.ones(self.batch_size) * params['relation_vocab']['DUMMY_START_RELATION']).long()
        self.entity_embedding_size = self.embedding_size

        if self.use_entity_embeddings:
            self.m = 4
        else:
            self.m = 2



        self.policy_step = Policy_step(m=self.m, embedding_size=self.embedding_size, hidden_size=self.hidden_size).to(self.device)
        self.policy_mlp = Policy_mlp(self.hidden_size, self.m, self.embedding_size).to(self.device)


        """
        不知道需不需要改
        # """

    def get_mem_shape(self):
        return (self.LSTM_Layers, 2, None, self.m * self.hidden_size)


    def action_encoder(self, next_relations, next_entities, next_logicals = None):
        entity_embedding = self.entity_embedding(next_entities)
        relation_embedding = self.relation_embedding(next_relations)

        action_embedding = relation_embedding

        if self.use_entity_embeddings:
            action_embedding = torch.cat([action_embedding, entity_embedding], dim=-1)

        return action_embedding

    def step(self, next_relations, next_entities, prev_state, prev_relation, query_embedding, current_entities,
              range_arr, first_step_of_test):

        # (original batch_size * num_rollout, （relation /+ logical /+ entity）* self.embedding_size)
        prev_action_embedding = self.action_encoder(prev_relation, current_entities)

        # 1. one step of rnn
        # input = prev_action_embedding, (h_0, c_0) = (state_emb[0], state_emb[1])
        output, new_state = self.policy_step(prev_action_embedding, prev_state)

        # Get state vector
        prev_entity = self.entity_embedding(current_entities)
        if self.use_entity_embeddings:
            # [batch_size * rollout , m* embedding_size + 2* embedding_size]
            state = torch.cat([output, prev_entity], dim=-1)
        else:
            # [batch_size * rollout, m * embedding_size]
            state = output

        # 候选action嵌入名单 [batch_size * num_rollout , action_space_size, m * embedding_size ]
        candidate_action_embeddings = self.action_encoder(next_relations, next_entities)
        # batchsize * rollout
        query_embedding = self.relation_embedding(query_embedding)

        state_query_concat = torch.cat([state, query_embedding], dim=-1)

        # MLP for policy#
        # [batch_size * num_rollout, m * embedding_size]
        output = self.policy_mlp(state_query_concat)
        # [original batch_size * num_rollout, 1, 2D], D=self.hidden_size
        output_expanded = torch.unsqueeze(output, dim=1)
        #[batch_size * num_rollout , action_space_size]
        prelim_scores = torch.sum(candidate_action_embeddings * output_expanded, dim=2)

        # Masking PAD actions
        comparison_relation = torch.ones_like(next_relations).int() * self.rPAD  # matrix to compare
        # comparison_logical = torch.ones_like(next_logicals).int() * self.lPAD
        # comparison_tensor = torch.cat([comparison_relation, comparison_logical], dim=-1)
        relation_mask = next_relations== comparison_relation  # The mask
        # logical_mask = next_logicals == comparison_logical
        # mask = relation_mask*1 == logical_mask*1
        # the base matrix to choose from if dummy relation
        dummy_scores = torch.ones_like(prelim_scores) * -99999.0
        # [original batch_size * num_rollout, max_num_actions]
        scores = torch.where(relation_mask, dummy_scores, prelim_scores)

        # 4 sample action
        action = torch.distributions.categorical.Categorical(logits=scores) # [original batch_size * num_rollout, 1]
        label_action = action.sample() # [original batch_size * num_rollout,]

        # loss
        # 5a.
        loss = torch.nn.CrossEntropyLoss(reduce=False)(scores, label_action)

        # 6. Map back to true id
        chosen_relations = next_relations[list(torch.stack([range_arr, label_action]))]

        return loss, new_state, F.log_softmax(scores), label_action, chosen_relations

    # def fill_query(self, prev_entities, prev_relations, prev_logicals, current_entities):

class EntityAgent(Agent):
    def __init__(self, params):
        Agent.__init__(self, params)

