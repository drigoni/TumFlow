import json

import numpy as np

#                        [H,C,N,O,F,P ,S ,Cl,Se,Br,I ]
cancer_atomic_num_list = [1, 6, 7, 8, 9, 15, 16, 17, 34, 35, 53, 0]  # 0 is for virtual node


def one_hot_cancer(data, out_size=80):
    num_max_id = len(cancer_atomic_num_list)
    assert data.shape[0] == out_size
    b = np.zeros((out_size, num_max_id), dtype=np.float32)
    for i in range(out_size):
        ind = cancer_atomic_num_list.index(data[i])
        b[i, ind] = 1.0
    return b


def transform_fn_cancer(data):
    node, adj, label = data
    node = one_hot_cancer(node).astype(np.float32)
    # single, double, triple and no-bond. Note that last channel axis is not connected instead of aromatic bond.
    adj = np.concatenate([adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)], axis=0).astype(np.float32)
    return node, adj, label


def get_val_ids():
    file_path = "./data/valid_idx_melanoma_skmel28.json"
    print("loading train/valid split information from: {}".format(file_path))
    with open(file_path) as json_data:
        data = json.load(json_data)
    val_ids = [idx - 1 for idx in data]
    return val_ids
