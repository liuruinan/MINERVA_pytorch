#!/usr/bin/env bash

data_input_dir="datasets/data_preprocessed/FB15K-237/"
vocab_dir="datasets/data_preprocessed/FB15K-237/vocab"
total_iterations=4000
path_length=3
hidden_size=50
embedding_size=50
batch_size=128
beta=0.2
Lambda=0.2
use_cluster_embeddings=0
use_entity_embeddings=0
train_entity_embeddings=0
train_relation_embeddings=1
base_output_dir="output/fb15k-237/"
load_model=0
model_load_dir="saved_models/fb15k-237"
nell_evaluation=0
