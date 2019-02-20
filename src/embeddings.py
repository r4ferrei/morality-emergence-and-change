import os
import pickle
import numpy as np

def load(
        dir='data/hamilton-historical-embeddings/sgns/',
        vocab_file='1990-vocab.pkl',
        embedding_file='1990-w.npy'):
    with open(os.path.join(dir, vocab_file), 'rb') as vocab_file:
        vocab = pickle.load(vocab_file)
        embeddings = np.load(os.path.join(dir, embedding_file))

        dic = {}
        for word, vec in zip(vocab, embeddings):
            if np.linalg.norm(vec) > 0: # embedding exists
                dic[word] = vec

        return dic
