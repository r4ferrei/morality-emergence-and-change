import numpy as np
import torch

DEVICE = 'cpu'

def build_loo_classification_dataset(emb_mats):
    '''
    Given embedding matrices for different categories, return a leave-one-out
    classification task dataset.

    Args:
        emb_mats: a list of embedding matrices, one matrix per class, where
        each matrix has N_i rows, where N_i is the size of the i-th class,
        by D columns, where D is the embedding dimension.

    Returns: a list of classification frames. Each classification frame is
    a dictionary with:
        - 'probe': a [D] tensor;
        - 'class': an integer class index corresponding to the probe;
        - 'emb_mats': a list of tensors corresponding to the embedding matrices,
            except that the probe is excluded from its corresponding class.
    '''

    emb_mats = [torch.tensor(m, device=DEVICE) for m in emb_mats]

    def drop_row(A, i):
        return torch.cat((A[:i], A[(i+1):]))

    res = []
    for i in range(len(emb_mats)):
        for j in range(len(emb_mats[i])):
            probe = emb_mats[i][j]
            class_ = i

            this_emb_mats = []
            for k in range(len(emb_mats)):
                this_emb_mats.append(
                        drop_row(emb_mats[k], j) if k == i
                        else emb_mats[k])

            res.append({
                'probe'    : probe,
                'class'    : class_,
                'emb_mats' : this_emb_mats
                })
    return res
