import nltk
import constant
import re
import embeddings
import pandas as pd
import os
from nltk.corpus import wordnet

def split_file_line(line):
    '''
    Process 
    '''
    split_line = re.split(r"[\t \n,]+", line)
    return list(filter(None, split_line))

def get_all_words(word, full_wordset):
    new_list = []
    if word not in full_wordset:
        for w in full_wordset:
            if w.startswith(word[:-1]) and len(w) - len(word) < 3:
                if len(wordnet.synsets(w)) > 0:
                    new_list.append(w)
    else:
        new_list.append(word)
    return new_list

def complete_word(word, seen_words, full_wordset):
    new_words = set([])
    all_words = get_all_words(word, full_wordset)
    ss = []
    targ_word = None
    while len(all_words) > 0:
        targ_word_i = all_words.index(min(all_words, key=len))
        targ_word = all_words.pop(targ_word_i)
        ss = wordnet.synsets(targ_word)
        if len(ss) > 0:
            break
    if len(ss) > 0:
        all_names = set([x.name() for x in ss[0].lemmas()[0].derivationally_related_forms()])
        all_names.add(targ_word)
        for name in all_names:
            if name in full_wordset and name not in seen_words:
                new_words.add(name)
    return new_words

def parse_dict(full_wordset):
    mfd = []
    seen_words = set([])
    with open(constant.DICTIONARY_PATH) as f:
        lines = f.readlines() # list containing lines of file
        for line in lines:
            l = split_file_line(line)
            categories = [int(x) for x in l[1:]]
            if 12 not in categories:
                if l[0][-1] == '*': # complete wildcard
                    new_words = complete_word(l[0], seen_words, full_wordset)
                elif l[0] not in seen_words:
                    new_words = set([l[0]])
                for category in categories:
                    for word in new_words:
                        mfd.append({'word':word, 'category':category})
                        seen_words.add(word)
    return mfd

_,vocab_list = embeddings.load_all(years=[1800], dir=constant.SGNS_DIR)
all_words_set = set(vocab_list)
mfd = parse_dict(all_words_set)
df = pd.DataFrame(mfd)

df.to_csv(os.path.join(constant.DATA_DIR,'cleaned_words.csv'))