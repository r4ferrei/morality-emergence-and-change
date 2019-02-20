import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

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

class ExemplarModel(nn.Module):
    '''
    Exemplar model of categorization.
    Categories are biased by the inverse of category size. In other words,
    this model computes the mean activation across exemplars per category,
    this de-biasing the given class sizes.

    Parameters:
        width: a single number representing the kernel width.

    Input:
        - [D] tensor embedding of probe word;
        - list of [N_i, D] embedding matrices for each class, i = 1, ..., C.

    Output:
        - [C] tensor of unnormalized class likelihoods.
    '''

    def __init__(self, metric='l2'):
        '''
        Args:
            metric: metric of vector distance, one of 'l2' or 'cosine'
        '''

        super().__init__()
        self.kernel_width = nn.Parameter(
                torch.tensor([1.], dtype=torch.float64, device=DEVICE))
        self.metric = metric

    def forward(self, probe, emb_mats):
        output = []
        for emb_mat in emb_mats:
            probe_repeated = probe.repeat(len(emb_mat), 1)
            if self.metric == 'l2':
                dists = F.pairwise_distance(probe_repeated, emb_mat)
            elif self.metric == 'cosine':
                dists = (1 - F.cosine_similarity(probe_repeated, emb_mat)) / 2
            else:
                raise ValueError("unknown distance metric '{}'".format(
                    self.metric))

            activation = torch.exp(-dists / self.kernel_width)
            activation = activation.mean()
            output.append(activation)

        return torch.stack(output)

def batch_nll_loss(model, batch):
    '''
    Computes the average negative log-likelihood loss over a classification batch.

    Args:
        model: an instance of `ExemplarModel`.
        batch: a list in the same shape as `build_loo_classification_dataset`.

    Returns: a single tensor with the average NLL loss over the batch.
    '''

    losses = []
    for instance in batch:
        lik = model(instance['probe'], instance['emb_mats'])
        class_lik = lik[instance['class']]
        total_lik = lik.sum()
        nll = -(torch.log(class_lik) - torch.log(total_lik))
        losses.append(nll)
    return torch.stack(losses).mean()

def train_model(dataset, lr=.001, batch_size=32, threshold=1e-6, patience=10):
    '''
    Trains an exemplar classification model given a dataset.

    Args:
        dataset: a dataset from `build_loo_classification_dataset`.
        lr: learning rate.
        batch_size: batch size for optimization.
        threshold: stop when loss decreases by less than this from best epoch.
        patience: wait until threshold is not exceeded by this many times.

    Returns: trained `ExemplarModel`.
    '''

    model = ExemplarModel()
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    min_loss = 1000
    curr_epoch = 0
    patience_counter = 0
    while True:
        random.shuffle(dataset)
        total_loss = 0
        num_batches = 0
        for start_idx in range(0, len(dataset), batch_size):
            opt.zero_grad()

            batch = dataset[start_idx : (start_idx+batch_size)]
            loss  = batch_nll_loss(model, batch)

            total_loss += float(loss.detach())
            num_batches += 1

            loss.backward()
            opt.step()

        avg_loss = total_loss / num_batches
        print("Epoch %d, average loss %f" % (curr_epoch, avg_loss))
        curr_epoch += 1

        if avg_loss >= min_loss - threshold:
            patience_counter += 1
        else:
            patience_counter = 0

        if patience_counter >= patience:
            break

        min_loss = min(min_loss, avg_loss)

    return model

def accuracy(model, dataset):
    '''
    Computes model classification accuracy over the given dataset.

    Args:
        model: an instance of `ExemplarModel`.
        dataset: a dataset like that from `build_loo_classification_dataset`.

    Returns: a single floating-point accuracy value.
    '''

    num_correct = 0
    num_total   = 0
    for instance in dataset:
        lik = model(instance['probe'], instance['emb_mats'])
        _, pred = torch.max(lik, dim=0)
        num_correct += bool((pred == instance['class']).detach())
        num_total += 1
    return num_correct / num_total
