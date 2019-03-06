import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
import copy

DEVICE = 'cpu'

def build_loo_classification_dataset(emb_mats, words_per_class):
    '''
    Given embedding matrices for different categories, return a leave-one-out
    classification task dataset.

    Args:
        emb_mats: a list of embedding matrices, one matrix per class, where
        each matrix has N_i rows, where N_i is the size of the i-th class,
        by D columns, where D is the embedding dimension.

        words_per_class: [C, N_i] list of string words per class, aligned
        with `emb_mats`.

    Returns: a list of classification frames. Each classification frame is
    a dictionary with:
        - 'probe': a [D] tensor;
        - 'probe_str' : string corresponding to probe;
        - 'class': an integer class index corresponding to the probe;
        - 'emb_mats': a list of tensors corresponding to the embedding matrices,
            except that the probe is excluded from its corresponding class.
        - 'words_per_class': same as the given `words_per_class`, except that
            the probe is dropped.
    '''

    emb_mats = [torch.tensor(m, device=DEVICE) for m in emb_mats]

    def drop_row(A, i):
        return torch.cat((A[:i], A[(i+1):]))

    def drop_word(W, i, j):
        V = copy.deepcopy(W)
        del V[i][j]
        return V

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
                'probe'           : probe,
                'probe_str'       : words_per_class[i][j],
                'class'           : class_,
                'emb_mats'        : this_emb_mats,
                'words_per_class' : drop_word(words_per_class, i, j),
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
        - [B, D] tensor embedding of probe word (B is the batch size);
        - [B, C] list of [N_i, D] embedding matrices where C is the number
            of classes and N_i is the size of the i-th class.

    Output:
        - [B, C] tensor of unnormalized class likelihoods.
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

    def probe_to_mat_dist(self, probe, emb_mat):
        '''
        Computes the distances between the probe and all embeddings.

        Args:
            probe: [B, D] tensor.
            emb_mat: [B, N, D] tensor.

        Returns: [B, N] tensor.
        '''

        b, n, d = emb_mat.shape
        assert((b, d) == probe.shape)

        probe_repeated = probe.unsqueeze(dim=1).repeat(1, n, 1)
        assert((b, n, d) == probe_repeated.shape)

        if self.metric == 'l2':
            A = probe_repeated.reshape(b*n, d)
            B = emb_mat.reshape(b*n, d)
            dists = F.pairwise_distance(A, B)
            dists = dists.reshape(b, n)
        elif self.metric == 'cosine':
            dists = (1 - F.cosine_similarity(
                probe_repeated, emb_mat, dim=2)) / 2
        else:
            raise ValueError("unknown distance metric '{}'".format(self.metric))

        assert((b, n) == dists.shape)
        return dists

    def forward(self, probes, emb_mats):
        # For each instance in batch, list of number of examples per class.
        class_sizes = [
                [E.shape[0] for E in embs]
                for embs in emb_mats
                ]

        emb_mats = [torch.cat(embs) for embs in emb_mats] # [B] of [N, D]
        emb_mats = torch.stack(emb_mats) # [B, N, D]

        b, _, d = emb_mats.shape
        assert((b, d) == probes.shape)

        dists = self.probe_to_mat_dist(probes, emb_mats)
        acts  = torch.exp(-dists / self.kernel_width)

        results = []
        for i in range(b):
            # [C] list of [N_i] tensors.
            per_class = torch.split(acts[i], class_sizes[i])
            activations = [A.mean() for A in per_class]
            results.append(torch.stack(activations))

        return torch.stack(results)

def batch_nll_loss(model, batch):
    '''
    Computes the average negative log-likelihood loss over a classification batch.

    Args:
        model: an instance of `ExemplarModel`.
        batch: a list in the same shape as `build_loo_classification_dataset`.

    Returns: a single tensor with the average NLL loss over the batch.
    '''

    probes = []
    emb_mats = []
    for instance in batch:
        probes.append(instance['probe'])
        emb_mats.append(instance['emb_mats'])
    probes = torch.stack(probes)

    unnorm_loglik = model(probes, emb_mats)
    b, c = unnorm_loglik.shape

    total_loglik = torch.sum(unnorm_loglik, dim=1)
    assert((b,) == total_loglik.shape)

    class_loglik = torch.stack(
            [unnorm_loglik[i, inst['class']] for i, inst in enumerate(batch)])
    assert((b,) == class_loglik.shape)

    nlls = -(torch.log(class_loglik) - torch.log(total_loglik))
    return nlls.mean()

def train_model(dataset, lr=.001, batch_size=32, threshold=1e-6, patience=15,
        metric='l2'):
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

    model = ExemplarModel(metric=metric)
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

        if curr_epoch % 10 == 0:
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

def true_loo_accuracy(emb_mats, words_per_class, metric='l2'):
    '''
    Given embedding matrices for different categories, train models and return
    leave-one-out model accuracy.

    This performs LOO at two levels: at the outer level, it leaves a test
    example out and trains on the rest (accuracy comes from here). At the
    inner level, the training is performed by optimizing inner LOO
    classification NLL.

    Args:
        emb_mats: list of embedding matrices, one per class, each represeting
        a list of embedding rows.

        words_per_class: list of strings for each class, aligned with
        `emb_mats`.

    Returns: floating-point LOO accuracy.
    '''

    outer_dataset = build_loo_classification_dataset(emb_mats, words_per_class)
    num_correct = 0
    for i, instance in enumerate(outer_dataset):
        print("Training on LOO iteration %d/%d" % (i, len(outer_dataset)))
        inner_dataset = build_loo_classification_dataset(
                instance['emb_mats'],
                instance['words_per_class'])
        model = train_model(inner_dataset, metric=metric)

        lik = model(
                instance['probe'].unsqueeze(dim=0),
                [instance['emb_mats']])
        b, c = lik.shape
        assert(b == 1)

        _, pred = torch.max(lik, dim=1)
        num_correct += bool((pred == instance['class']).detach())

    return num_correct / len(outer_dataset)

def centroid_loo_classification(emb_mats, words_per_class):
    '''
    Given embedding matrices for different categories, return leave-one-out
    classification results of centroid classifier using L2 distance metric.

    Args:
        emb_mats: list of embedding matrices, one per class, each representing
        a list of embedding rows.

        words_per_class: list of strings per class, aligned with `emb_mats`.

    Returns: DataFrame with columns 'instance', 'true_class', 'predicted_class'.
    '''

    dataset = build_loo_classification_dataset(emb_mats, words_per_class)
    results = []

    num_correct = 0

    for instance in dataset:
        centroids = [torch.mean(embs, dim=0) for embs in instance['emb_mats']]
        centroids = torch.stack(centroids)

        probe_repeated = instance['probe'].repeat(centroids.shape[0], 1)
        dists = F.pairwise_distance(probe_repeated, centroids)
        assert(len(dists.shape) == 1)

        num_correct += bool((torch.argmin(dists) == instance['class']).detach())

        pred = torch.argmin(dists).detach()
        true_ = instance['class']

        results.append({
            'instance'        : str(instance['probe_str']),
            'true_class'      : int(true_),
            'predicted_class' : int(pred),
            })

    print("acc: {}".format(num_correct / len(dataset)))

    return pd.DataFrame(results)
