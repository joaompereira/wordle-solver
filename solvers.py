import numpy as np
from time import time
import itertools as itt
import pickle
from compiler_options import compiler_decorator, prange, generated_jit

def word_checker(test_word, true_word):
    test_word = remove_accents(np.array((test_word,))).view(np.int32)
    true_word = remove_accents(np.array((true_word,))).view(np.int32)
    return word_checker_inner(test_word, true_word)

@compiler_decorator()
def word_checker_inner(test_word, true_word):

    test_checked = test_word == true_word
    test = 2*test_checked

    true_checked = test_checked.copy()

    for i in range(5):
        if not test_checked[i]:
           for j in range(5):
               if not true_checked[j] and test_word[i]==true_word[j]:
                   test[i] = 1
                   true_checked[j] = True

    return test

class KeyDict(dict):
    def __missing__(self, key):
        return key


def remove_accents(words):
    accent_dictionary = KeyDict(
        á='a', â='a', ã='a', ç='c', è='e', é='e',
        ê='e', í='i', ï='i', ó='o', ô='o', õ='o',
        ú='u', û='u', ü='u'
    )
    letters, indices = np.unique(words.view('U1'), return_inverse=True)
    letters_na = np.array([accent_dictionary[k] for k in letters])
    return letters_na[indices].view('U5')

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

@compiler_decorator(parallel=True)
def max_row(M):
    a = M.shape[0]
    res = np.zeros(a, dtype=M.dtype)
    for i in prange(a):
        res[i] = M[i].max()

    return res



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


class MinMaxSolver(WordleSolver):
    def __init__(self, words, n_bestwords):
        super(MinMaxSolver, self).__init__(words, n_bestwords)
        self.table = word_table(words, n_bestwords)
        #cb = np.full((self.n_bestwords,), True)
        buckets = word_buckets(self.table)
        self.first_word_ind = np.max(buckets, axis=1).argmin()
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
            log_bag = np.zeros((self.n_words))
            for j in np.nonzero(self.still)[0]:
                buckets = word_buckets(self.table, self.cb[j])
                log_bag += -np.log(np.max(buckets, axis=1))
                log_bag[:self.n_bestwords][self.cb[j]] += (np.log(scb[j]) / scb[j])

            guess_ind = log_bag.argmax()

        self.last_guess_ind = guess_ind
        return self.words[guess_ind]

class SolutionMatrixSolver(WordleSolver):
    def __init__(self, words, n_bestwords, solution_matrix=None):
        super().__init__(words, n_bestwords)
        self.table = word_table(words, n_bestwords)
        if type(solution_matrix) is np.ndarray:
            self.solution_matrix = solution_matrix
        elif type(solution_matrix) is str:
            with open('solvers/' + solution_matrix + '.sm', 'rb') as f:
                self.solution_matrix = pickle.load(f)
        elif solution_matrix is None:
            raise TypeError('SolutionMatrixSolver() missing required argument \'solution_matrix\' (pos 2)')
        else:
            raise NotImplementedError
        self.first_word_ind = self.solution_matrix[0,0]
        self.first_word = self.words[self.first_word_ind]

    def start(self, n_words=1):
        if n_words > 1:
            raise NotImplementedError
        self.cb = np.full(self.n_bestwords, True)
        self.last_guess_ind = self.first_word_ind
        self.level = 0
        return self.first_word

    def guess(self, answer_2_previous):
        self.level += 1
        answer_short = np.matmul(answer_2_previous, 3**np.arange(5)[::-1])

        self.cb[self.cb] = self.table[self.last_guess_ind, self.cb] == answer_short

        j = np.nonzero(self.cb)[0][0]
        guess_ind = self.solution_matrix[j, self.level]
        self.last_guess_ind = guess_ind
        return self.words[guess_ind]

@compiler_decorator()
def get_minexpt_order_entropy(table, col_subset, row_subset):
    nw, nwb = table.shape
    buckets = word_buckets(table[:, col_subset][row_subset])
    bmorethanone = np.count_nonzero(buckets, axis=1) > 1
    row_subset[row_subset] = bmorethanone
    buckets = buckets[bmorethanone]
    scb = np.count_nonzero(col_subset)
    entropy = np.zeros(nw)
    entropy[row_subset] = -calculate_entropy(buckets / scb)
    entropy[:nwb][col_subset] -= np.log(scb) / scb
    ind_ord = entropy.argsort()

    return ind_ord

@compiler_decorator()
def min_expected_tries(table, max_words=np.zeros(10), E_max = 10000,
                       row_subset=np.full(0, True), col_subset = np.full(0, True),
                       level=0, solution_matrix = np.zeros((0,0), dtype=np.int64),
                       print_level = 1):
    nw, nwb = table.shape
    if row_subset.shape[0] == 0:
        row_subset = np.full(nw, True)
    if col_subset.shape[0] == 0:
        col_subset = np.full(nwb, True)
    row_subset = np.copy(row_subset)
    if solution_matrix.shape[0]==0:
        solution_matrix = np.full((nwb, 10), -1)

    solution_matrix_c = solution_matrix.copy()
    scb = np.count_nonzero(col_subset)
    E = E_max


    ind_ord = get_minexpt_order_entropy(table, col_subset, row_subset)
    max_words_ = np.count_nonzero(row_subset)
    if max_words[0] > 0:
        max_words_ = min(max_words_, max_words[0])
    for i_ in range(max_words_):
        if E <= 2 * scb - 2:
            return E, solution_matrix
        elif (E <= 2 * scb) and i_ > 0:
            return E, solution_matrix
        i = ind_ord[i_]
        E_try = scb
        #tics = table[i][col_subset]
        nzss = np.zeros(243, dtype=np.int64)
        subsets = np.full((243, nwb), False)
        nzssgt0 = 0
        for k in np.nonzero(col_subset)[0]:
            j = table[i, k]
            subsets[j, k] = True
            nzss[j] += 1
            if nzss[j] == 1:
                nzssgt0 += 1
        at_least = 2*scb - nzssgt0 - nzss[242]
        for j in range(242):
            if nzss[j] == 0:
                continue
            at_least -= 2 * nzss[j] - 1
            if nzss[j] <= 2:
                E_try += 2 * nzss[j] - 1
                inds = np.nonzero(subsets[j])[0]
                solution_matrix_c[inds[0], level + 1] = inds[0]
                if nzss[j] == 2:
                    solution_matrix_c[inds[1], level + 1] = inds[0]
                    solution_matrix_c[inds[1], level + 2] = inds[1]
            else:
                E_try += min_expected_tries(table, max_words[1:], E - E_try - at_least,
                                            row_subset, subsets[j], level + 1, solution_matrix_c)[0]

            if E_try + at_least >= E:
                E_try = E
                break

        if E_try < E:
            E = E_try
            for j in col_subset.nonzero()[0]:
                solution_matrix[j, level+1:] = solution_matrix_c[j, level+1:]
                solution_matrix[j, level] = i
            if level <= print_level:
                print(level, i_, scb, E, i)

    return E, solution_matrix