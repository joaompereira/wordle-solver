import numpy as np
from time import time
import itertools as itt
from main import word_checker_inner, remove_accents
from compiler_options import compiler_decorator, prange

class WordleSolver():
    def __init__(self, words, n_bestwords):
        self.n_bestwords = n_bestwords
        self.words = words
        self.n_words = len(self.words)

    def start(self, n_words):
        # Start the algorithm
        # n_words: number of words in challenge
        #     Examples    1: termooo, 2: dueto, 4: quarteto
        # Returns your favorite starting word
        raise NotImplementedError

    def guess(self, answer_2_previous):
        # Gets the answer from the previous guess
        # Provides a new guess
        raise NotImplementedError


def word_table(words, words_before):
    return word_table_inner(remove_accents(words).view(int).reshape(-1, 5), words_before)

@compiler_decorator(parallel=True)
def word_table_inner(lines, words_before):
    nw = lines.shape[0]
    table = np.zeros((nw, words_before), dtype=np.ubyte)

    for i in range(nw):
        for j in range(words_before):
            v = word_checker_inner(lines[i], lines[j])
            s = v[0]
            for k in range(1, 5):
                s = 3 * s + v[k]
            table[i, j] = s

    return table

@compiler_decorator(parallel=True)
def word_buckets(table, subset=None):
    nw, nw2 = table.shape
    word_buckets = np.empty((nw,243))

    if subset is None:
        for i in range(nw):
            word_buckets[i, :] = np.bincount(table[i, :], minlength=243)
    else:
        for i in range(nw):
            v = table[i, :]
            word_buckets[i, :] = np.bincount(v[subset], minlength=243)

    return word_buckets

@compiler_decorator(parallel=True)
def calculate_entropy(word_buckets_scaled):
    nw = word_buckets_scaled.shape[0]
    word_entropy = np.empty(nw)

    for i in prange(nw):
        v = word_buckets_scaled[i, :]
        v = v[v>0]
        word_entropy[i] = -np.dot(v, np.log(v))

    return word_entropy

class EntropySolver(WordleSolver):
    def __init__(self, words, n_bestwords):
        super(EntropySolver, self).__init__(words, n_bestwords)
        self.table = word_table(words, n_bestwords)
        #cb = np.full((self.n_bestwords,), True)
        buckets = word_buckets(self.table)
        self.first_word_ind = calculate_entropy(buckets / self.n_bestwords).argmax()
        self.first_word = self.words[self.first_word_ind]

    def start(self, n_words=1):
        self.cb = np.full((n_words, self.n_bestwords), True)
        self.still = np.full((n_words,), True)
        self.last_guess_ind = self.first_word_ind
        return self.first_word

    def guess(self, answer_2_previous):
        answer_short = np.matmul(answer_2_previous, 3**np.arange(5)[::-1])
        self.still[answer_short == 242] = False

        for j in np.nonzero(self.still)[0]:
            self.cb[j][self.cb[j]] = (self.table[self.last_guess_ind, self.cb[j]] == answer_short[j])

        scb = np.count_nonzero(self.cb, axis=1)
        found_word = (scb==1) & self.still
        if np.any(found_word):
            j = np.nonzero(found_word)[0][0]
            guess_ind = np.nonzero(self.cb[j])[0][0]
        else:
            entropy = np.zeros((self.n_words))
            for j in np.nonzero(self.still)[0]:
                buckets = word_buckets(self.table, self.cb[j])
                entropy += calculate_entropy(buckets / scb[j])
                entropy[:self.n_bestwords][self.cb[j]] += (np.log(scb[j]) / scb[j])

            guess_ind = entropy.argmax()

        self.last_guess_ind = guess_ind
        return self.words[guess_ind]

