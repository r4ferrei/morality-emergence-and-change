import os
import pickle
import numpy as np

def load(
        dir='data/hamilton-historical-embeddings/sgns/',
        vocab_file='1990-vocab.pkl',
        embedding_file='1990-w.npy'):
    '''
    Loads embeddings in given directory with given filenames.
    Skips zero embeddings.

    Return: dictionary of word -> numpy embedding.
    '''

    with open(os.path.join(dir, vocab_file), 'rb') as vocab_file:
        vocab = pickle.load(vocab_file)
        embeddings = np.load(os.path.join(dir, embedding_file))

        dic = {}
        for word, vec in zip(vocab, embeddings):
            if np.linalg.norm(vec) > 0: # embedding exists
                dic[word] = vec

        return dic

def load_all(
        dir='data/hamilton-historical-embeddings/sgns/',
        years=list(range(1800, 1991, 10))):
    '''
    Loads embeddings for all decades and produces a conjunctive vocabulary.

    Args:
        dir: embeddings directory.
        years: list of years of embeddings to load.

    Returns: tuple (embeddings, vocab), where `embeddings` is a dictionary
    of year -> <embedding dict> (see `load`), and `vocab` is a list of words
    that have (non-zero) embeddings in all years.
    '''

    vocab = None
    res = {}
    for year in years:
        vocab_file = '%d-vocab.pkl' % year
        embedding_file = '%d-w.npy' % year
        embs = load(dir, vocab_file, embedding_file)

        res[year] = embs

        this_vocab = set(embs.keys())
        if vocab is None:
            vocab = this_vocab
        else:
            vocab = vocab & this_vocab

    return res, list(vocab)

def convert_words_to_embedding_matrix(words, embs):
    '''
    Converts a list of words to an embedding matrix.

    Args:
        words: a list of words of length N.
        embs: an embedding dictionary (see `load`) of dimension D.

    Returns: an [N, D] ndarray containing embeddings.

    Raises an exception if some embedding is not found.
    '''

    n = len(words)
    d = len(next(iter(embs.values())))
    res = np.zeros((n, d))
    for i, word in enumerate(words):
        try:
            res[i] = embs[word]
            assert(np.linalg.norm(res[i]) > 0)
        except:
            raise ValueError("no embedding for '%s'" % word)
    return res
