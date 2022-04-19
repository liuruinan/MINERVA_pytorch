#!/usr/bin/env bash

data_input_dir="datasets/data_preprocessed/WN18RR/"
vocab_dir="datasets/data_preprocessed/WN18RR/vocab"
total_iterations=1000
path_length=3
hidden_size=50
embedding_size=50
batch_size=256
beta=0.06
Lambda=0.00
use_entity_embeddings=0
use_cluster_embeddings=0
train_entity_embeddings=0
train_relation_embeddings=1
base_output_dir="output/WN18RR/"
load_model=0
model_load_dir="saved_models/WN18RR"
nell_evaluation=0
