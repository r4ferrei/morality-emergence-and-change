import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
import copy
import embeddings
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os
import pickle
import collections

DEVICE = 'cuda'

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
    Kernel density estimator model of categorization using Gaussian kernel.

    Parameters:
        width: a single number representing the kernel width.

    Input:
        - [B, D] tensor embedding of probe word (B is the batch size);
        - [B, C] list of [N_i, D] embedding matrices where C is the number
            of classes and N_i is the size of the i-th class.

    Output:
        - [B, C] tensor of unnormalized class likelihoods.
    '''

    def __init__(self, metric='l2', width=.15):
        '''
        Args:
            metric: metric of vector distance, one of 'l2' or 'cosine'
            width: initial kernel width value.
        '''

        super().__init__()
        self.kernel_width = nn.Parameter(
                torch.tensor([width], dtype=torch.float64, device=DEVICE))
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
            assert(False)
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
        # Technically the width should be squared, but since we cancel out the
        # outer normalization constant anyway, it doesn't make a difference and
        # this is simpler.
        acts  = torch.exp(-torch.pow(dists, 2) / self.kernel_width)

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

def train_model(dataset, lr=.005, batch_size=64, threshold=1e-6, patience=15,
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

def kernel_loo_classification(emb_mats, words_per_class,
        seeds_for_fda, embs, metric='l2',
        kernel_width=None):
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

        kernel_width: if given, use pre-trained width for prediction.

    Returns: DataFrame with columns 'instance', 'true_class', 'predicted_class',
        and 'kernel_width'.
    '''

    try:
        assert(seeds_for_fda is None)
        assert(embs is None)
    except:
        raise NotImplementedError()

    results = []

    outer_dataset = build_loo_classification_dataset(emb_mats, words_per_class)
    num_correct = 0
    for i, instance in enumerate(outer_dataset):
        print("Training on LOO iteration %d/%d" % (i, len(outer_dataset)))
        inner_dataset = build_loo_classification_dataset(
                instance['emb_mats'],
                instance['words_per_class'])

        if kernel_width:
            model = ExemplarModel(metric='l2', width=kernel_width)
        else:
            model = train_model(inner_dataset, metric=metric)

        lik = model(
                instance['probe'].unsqueeze(dim=0),
                [instance['emb_mats']])
        b, c = lik.shape
        assert(b == 1)

        _, pred = torch.max(lik, dim=1)
        num_correct += bool((pred == instance['class']).detach())

        results.append({
            'instance'        : str(instance['probe_str']),
            'true_class'      : int(instance['class']),
            'predicted_class' : int(pred.detach()),
            'kernel_width'    : float(model.kernel_width.detach()),
            })

    print("acc: {}".format(num_correct / len(outer_dataset)))

    return pd.DataFrame(results)

#fda_cache = {}
#fda_filename = 'fda_cache.pkl'
#
#def load_fda_cache():
#    global fda_cache
#    try:
#        with open(fda_filename, 'rb') as f:
#            fda_cache = pickle.load(f)
#    except FileNotFoundError:
#        pass
#
#def persist_fda_cache():
#    global fda_cache
#    with open(fda_filename, 'wb') as f:
#        pickle.dump(fda_cache, f)

def cached_fit_fda(X, y):
    fda = LinearDiscriminantAnalysis(solver='eigen', shrinkage=.5,
            n_components=9)
            #n_components=10)
    fda.fit(X, y)
    return fda

#    key = (np.array(X).tobytes(), np.array(y).tobytes())
#    if key not in fda_cache:
#        fda = LinearDiscriminantAnalysis(solver='eigen', shrinkage=.5)
#        fda.fit(X, y)
#        fda_cache[key] = fda
#    return fda_cache[key]

def centroid_loo_classification(emb_mats, words_per_class,
        seeds_for_fda, embs):
    '''
    Given embedding matrices for different categories, return leave-one-out
    classification results of centroid classifier using L2 distance metric.

    Args:
        emb_mats: list of embedding matrices, one per class, each representing
        a list of embedding rows.

        words_per_class: list of strings per class, aligned with `emb_mats`.

    Returns: DataFrame with columns 'instance', 'true_class', 'predicted_class'.
    '''

    def fit_fda(probe_str_to_block):
        X = []
        y = []

        found_probe = False
        for i, words in enumerate(seeds_for_fda):
            emb_mat = embeddings.convert_words_to_embedding_matrix(words, embs)

            for word, vec in zip(words, emb_mat):
                if word == probe_str_to_block:
                    found_probe = True

                    # TODO wrong, just for testing!
                    #X.append(vec)
                    #y.append(i)
                else:
                    X.append(vec)
                    y.append(i)

        assert(found_probe)

        fda = cached_fit_fda(X, y)
        return fda

    def fda_transform(instance):
        instance['emb_mats'] = [
                fda.transform(emb_mat) for emb_mat in instance['emb_mats']]
        instance['emb_mats'] = [
                torch.tensor(m, device=DEVICE) for m in instance['emb_mats']]

        instance['probe'] = fda.transform(instance['probe'].reshape(1, -1))[0]
        instance['probe'] = torch.tensor(instance['probe'], device=DEVICE)

    dataset = build_loo_classification_dataset(emb_mats, words_per_class)
    results = []

    num_correct = 0

    #load_fda_cache()

    for instance in dataset:
        if seeds_for_fda:
            fda = fit_fda(instance['probe_str'])
            fda_transform(instance)

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

    #persist_fda_cache()

    print("acc: {}".format(num_correct / len(dataset)))

    return pd.DataFrame(results)

def matrix_centroid(emb_mat):
    return torch.mean(emb_mat, dim=0)

def probe_to_centroids_activations(probe, centroids):
    probe_repeated = probe.repeat(len(centroids), 1)
    dists = F.pairwise_distance(probe_repeated, centroids)
    assert(len(dists.shape) == 1)

    return torch.exp(-dists)

def centroid_tiered(emb_mats, words_per_class, pos_cats, neg_cats):
    dataset = build_loo_classification_dataset(emb_mats, words_per_class)

    num_correct = 0

    results = []

    for instance in dataset:
        # Embedding matrices for each category belonging to each polarity.
        pos_cat_mats = []
        neg_cat_mats = []

        for i, emb_mat in enumerate(instance['emb_mats']):
            if i in pos_cats:
                pos_cat_mats.append(emb_mat)
            elif i in neg_cats:
                neg_cat_mats.append(emb_mat)
            else:
                assert(False)

        # Embedding matrices for each polarity.
        pos_mat = torch.cat(pos_cat_mats)
        neg_mat = torch.cat(neg_cat_mats)

        pos_centroid = matrix_centroid(pos_mat)
        neg_centroid = matrix_centroid(neg_mat)

        pos_cat_centroids = [matrix_centroid(e) for e in pos_cat_mats]
        neg_cat_centroids = [matrix_centroid(e) for e in neg_cat_mats]

        all_centroids = (
                [pos_centroid, neg_centroid] +
                pos_cat_centroids + neg_cat_centroids
                )
        all_centroids = torch.stack(all_centroids)
        all_acts = probe_to_centroids_activations(
                instance['probe'], all_centroids)

        index = 0

        pos_act = all_acts[index]
        index += 1

        neg_act = all_acts[index]
        index += 1

        pos_acts = []
        for i in range(index, index+len(pos_cat_centroids)):
            pos_acts.append(all_acts[i])
        index += len(pos_cat_centroids)

        neg_acts = []
        for i in range(index, index+len(neg_cat_centroids)):
            neg_acts.append(all_acts[i])
        index += len(neg_cat_centroids)

        assert(index == len(all_acts))

        class_probs = [float('nan')] * (len(pos_cats) + len(neg_cats))

        top_norm_factor = pos_act + neg_act
        pos_norm_factor = sum(pos_acts)
        neg_norm_factor = sum(neg_acts)

        for cat in pos_cats + neg_cats:
            if cat in pos_cats:
                class_probs[cat] = (
                        pos_act / top_norm_factor *
                        pos_acts[pos_cats.index(cat)] / pos_norm_factor
                        )
            else:
                class_probs[cat] = (
                        neg_act / top_norm_factor *
                        neg_acts[neg_cats.index(cat)] / neg_norm_factor
                        )

        best_prob = -1
        best_class = -1
        for i in range(len(class_probs)):
            assert(not np.isnan(class_probs[i]))
            if class_probs[i] > best_prob:
                best_prob = class_probs[i]
                best_class = i

        num_correct += (best_class == instance['class'])

        results.append({
            'instance'        : str(instance['probe_str']),
            'true_class'      : instance['class'],
            'predicted_class' : best_class
            })

    print("acc: {}".format(num_correct / len(dataset)))

    return pd.DataFrame(results)

def knn_loo_classification(emb_mats, words_per_class, k=15):
    '''
    Given embedding matrices for different categories, return leave-one-out
    classification results of kNN classifier using L2 distance metric.

    Args:
        emb_mats: list of embedding matrices, one per class, each representing
        a list of embedding rows.

        words_per_class: list of strings per class, aligned with `emb_mats`.

        k: number of neighbours to compare to.

    Returns: DataFrame with columns 'instance', 'true_class', 'predicted_class'.
    '''

    dataset = build_loo_classification_dataset(emb_mats, words_per_class)
    results = []

    num_correct = 0

    for instance in dataset:
        dists_labels = []

        for i, emb_mat in enumerate(instance['emb_mats']):
            probe_repeated = instance['probe'].repeat(len(emb_mat), 1)
            dists = F.pairwise_distance(probe_repeated, emb_mat)
            assert(len(dists.shape) == 1)

            for dist in dists.detach().cpu():
                dists_labels.append((float(dist), i))

        dists_labels.sort()
        dists_labels = dists_labels[:k]
        counter = collections.Counter([lab for _, lab in dists_labels])

        pred = counter.most_common(1)[0][0]
        true_ = instance['class']
        num_correct += (pred == true_)

        results.append({
            'instance'        : str(instance['probe_str']),
            'true_class'      : int(true_),
            'predicted_class' : int(pred),
            })

    print("acc: {}".format(num_correct / len(dataset)))

    return pd.DataFrame(results)
