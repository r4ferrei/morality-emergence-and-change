import os
import pickle
import numpy as np
import scipy.io as sio
import pandas as pd
from nltk.corpus import stopwords
import constant

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

def load_all_nyt(
        dir='data/nyt'):
    base_year = 1990
    word_hash_df = pd.read_csv(os.path.join(dir, 'wordlist.csv'))
    assert len(list(word_hash_df.iterrows())) == 20000
    dic = {}
    for i in range(20):
        embeddings = {}
        mat_contents = sio.loadmat(os.path.join(dir, 'embeddings%d.mat' % i))
        embd_arr = mat_contents['U_%d' % i]
        for index, row in word_hash_df.iterrows():
            embeddings[row['word']] = embd_arr[index]
        dic[base_year+i] = embeddings
    return dic, list(word_hash_df['word'].values)

def load_all_fiction(dir):
    f = open(os.path.join(dir, 'word_vectors.txt'), "rb")
    emb_dict_all = {}
    all_words = set([])
    for line in f:
        line_arr = line.split()
        word, year, vec = line_arr[0].decode("utf-8") , int(line_arr[1]), [float(x) for x in line_arr[2:]]
        if year not in emb_dict_all:
            emb_dict_all[year] = {}
        all_words.add(word)
        emb_dict_yr = emb_dict_all[year]
        emb_dict_yr[word] = vec
    return emb_dict_all, list(all_words)

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

def choose_emb_dict(switch):
    if switch == 'NGRAM':
        emb_dict_all,vocab_list = load_all(dir=constant.SGNS_DIR, years=constant.ALL_YEARS)
    elif switch == 'FICTION':
        emb_dict_all,vocab_list = load_all_fiction(dir='D:/WordEmbeddings/kim')
    else:
        emb_dict_all,vocab_list = load_all_nyt(dir=constant.SGNS_NYT_DIR)
    return emb_dict_all,vocab_list

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

def get_sent_embed(emb_dict, phrase):
    if emb_dict is None:
        return None
    vocab = set(emb_dict.keys())
    stop_words = set(stopwords.words('english'))
    word_list = [x.lower() for x in phrase.split()]
    word_list = [emb_dict[x] for x in word_list if x not in stop_words and x in vocab]
    if len(word_list) > 0:
        return np.mean(word_list, axis=0)
    return None