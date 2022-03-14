import numpy as np
from time import time
import itertools as itt
from compiler_options import compiler_decorator
import solvers

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

def word_loader(string):
    with open(string + '.txt', encoding='utf8') as f:
        lines = f.read().splitlines()

    words_before = lines.index('-----')
    lines = np.array(lines[:words_before] + lines[words_before+1:])

    return lines, words_before

class WordleTester():
    def __init__(self, words_file):
        with open('words_datasets/' + words_file + '.txt', encoding='utf8') as f:
            lines = f.read().splitlines()

        self.n_bestwords = lines.index('-----')
        self.words = np.array(lines[:self.n_bestwords] + lines[self.n_bestwords + 1:])
        self.n_words = len(self.words)
        self.rng = np.random.default_rng()

    def tester(self, algorithm_class, n_words = 1, n_test_words = 100, maxtries=0):
        test_word_inds = self.rng.choice(self.n_bestwords, (n_test_words, n_words))
        timer = time()
        algorithm = algorithm_class(self.words, self.n_bestwords)
        setup_time = time() - timer
        tries = np.empty((n_test_words,), dtype=np.int32)
        timer = time()
        for k in range(n_test_words):
            tries[k] = self.test_algorithm_word(algorithm, test_word_inds[k, :], n_words, maxtries)
        avg_time = time() - timer

        return setup_time, tries, avg_time / n_test_words



    def test_algorithm_word(self, algorithm, true_word_inds, n_words = 1, maxtries=0):
        still = np.full((n_words,), True)
        guess = algorithm.start(n_words)
        answer = np.full((n_words, 5), 2)

        if maxtries == 0:
            generator = itt.count(1)
        else:
            generator = range(1, maxtries)

        for tries in generator:
            for k in np.argwhere(still):
                answer[k, :] = word_checker(guess, self.words[true_word_inds[k]])
                if np.all(answer[k, :] == 2):
                    still[k] = False

            if not np.any(still):
                break

            guess = algorithm.guess(answer)

        return tries

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

def print_tester_output(title, n_words, setup_time, tries, avg_time):
    print(f'## {title.upper()} ##')
    print(f' setup_time: {setup_time}')
    tries_bins = np.bincount(tries)
    print(' tries')
    for k in range(n_words, tries_bins.shape[0]):
        print(f'  {k}: {tries_bins[k]}')
    print(f' avg time per word: {avg_time}')

if __name__ == '__main__':
    tester = WordleTester('wordle')
    print_tester_output('wordle', 1, *tester.tester(solvers.EntropySolver, n_words=1, n_test_words=100))
    tester = WordleTester('palavras')
    print_tester_output('termooo', 1, *tester.tester(solvers.EntropySolver, n_words=1, n_test_words=100))
    print_tester_output('dueto', 2, *tester.tester(solvers.EntropySolver, n_words=2, n_test_words=100))
    print_tester_output('quarteto', 4, *tester.tester(solvers.EntropySolver, n_words=4, n_test_words=100))
    print('Bye')
