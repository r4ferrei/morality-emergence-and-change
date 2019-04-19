import embeddings

YEAR = 1990

all_vocabs = []

# N-grams
_, vocab = embeddings.load_all(
        dir   = 'data/hamilton-historical-embeddings/sgns/',
        years = [YEAR])
all_vocabs.append(vocab)
print("N-grams vocabulary contains %d words" % len(vocab))

# COHA
_, vocab = embeddings.load_all(
        dir   = 'data/coha-historical-embeddings/sgns/',
        years = [YEAR])
all_vocabs.append(vocab)
print("COHA vocabulary contains %d words" % len(vocab))

# NYT
_, vocab = embeddings.load_all_nyt(
        dir   = 'data/nyt/') # note: constant vocabulary through timw
all_vocabs.append(vocab)
print("NYT vocabulary contains %d words" % len(vocab))

all_vocabs = [set(v) for v in all_vocabs]
res = all_vocabs[0]
for v in all_vocabs:
    res = res & v

res = list(res)
print("Final vocabulary for %d contains %d words." % (YEAR, len(res)))
