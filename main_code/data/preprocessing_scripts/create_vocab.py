import json
import csv
import argparse
import os
import pickle
from collections import defaultdict


def entity_level_vocab():
    root_dir = '../../../'
    vocab_dir = root_dir+'datasets/data/NELL-betae/vocab/'
    dir = root_dir+'datasets/data/NELL-betae/'

    # entity2id = pickle.load(open(dir+'ent2id.pkl', 'rb'))
    # # print(entity2id['/m/01f38z'])
    # for line in entity2id:
    #     print(line.split()[0])

    entity_vocab = {}
    relation_vocab = {}

    '''
    NO_OP: The model takes no operation but remains on the same entity. 自旋
    UNK: When an entity is not seen during training but only during test, it doesn't have an embedding so it defaults to UNK (unknown) 未知
    DUMMY_START_RELATION: The embedding of an edge is computed as a composition of the relation + the entity it leads to. In the case of the first entity, 
                          there is no relation leading to it. So we assign a standard embedding called dummy_start which is assigned to 0. 初始化
    PAD: as we store our graph in a MxM matrix, several entities might not have M outgoing edges. 
         In that case, we assign the remaining edges as (entity, PAD, PAD). If the model takes the PAD relations, know something has gone wrong :)
    '''

    entity_vocab['PAD'] = len(entity_vocab)
    entity_vocab['UNK'] = len(entity_vocab)
    relation_vocab['PAD'] = len(relation_vocab)
    relation_vocab['DUMMY_START_RELATION'] = len(relation_vocab)
    relation_vocab['NO_OP'] = len(relation_vocab)
    relation_vocab['UNK'] = len(relation_vocab)


    entity_counter = len(entity_vocab)
    relation_counter = len(relation_vocab)

    if os.path.isfile(dir + 'full_graph.txt'):
        fact_files = ['full_graph.txt']
        print("Contains full graph")
    else:
        # fact_files = ['train.txt', 'dev.txt', 'test.txt', 'graph.txt']
        fact_files = ['train.txt', 'valid.txt', 'test.txt']


    for f in fact_files:
        with open(dir+f) as raw_file:
            csv_file = csv.reader(raw_file, delimiter='\t')
            for line in csv_file:

                e1, r, e2 = line

                if e1 not in entity_vocab:
                    entity_vocab[e1] = entity_counter
                    entity_counter += 1
                if e2 not in entity_vocab:
                    entity_vocab[e2] = entity_counter
                    entity_counter += 1
                if r not in relation_vocab:
                    relation_vocab[r] = relation_counter
                    relation_counter += 1

    with open(vocab_dir + 'entity_vocab.json', 'w+') as fout:
        json.dump(entity_vocab, fout)

    with open(vocab_dir + 'relation_vocab.json', 'w+') as fout:
        json.dump(relation_vocab, fout)


entity_level_vocab()

def logical_level_vocab():
    root_dir = '../../../'
    vocab_dir = root_dir + 'datasets/data_preprocessed/FB15K-237/vocab/'
    dir = root_dir + 'datasets/data_preprocessed/FB15K-237'

    logical_vocab = {}

    logical_vocab['PAD'] = len(logical_vocab)
    logical_vocab['UNK'] = len(logical_vocab)
    # logical_vocab['DUMMY_START_RELATION'] = len(logical_vocab)
    # logical_vocab['NO_OP'] = len(logical_vocab)
    logical_counter = len(logical_vocab)
    logical2id = pickle.load(open(os.path.join(dir, 'ent2id.pkl'), 'rb'))

    for key in logical2id:
        logical_vocab[key] = logical_counter
        logical_counter += 1

    with open(vocab_dir + 'entity_vocab.json', 'w+') as fout:
        json.dump(logical_vocab, fout)

logical_level_vocab()



# entity_level_vocab()
# cluster_level_vocab()