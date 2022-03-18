import numpy as np
from time import time
import itertools as itt
from compiler_options import compiler_decorator
from solvers import word_checker
import solvers

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

    def tester(self, algorithm=None, algorithm_class=None, n_words = 1, n_test_words = 100, maxtries=0):
        test_word_inds = self.rng.choice(self.n_bestwords, (n_test_words, n_words), replace=(n_words != 1))
        timer = time()
        algorithm = self.__get_algorithm(algorithm, algorithm_class)
        setup_time = time() - timer
        tries = np.empty((n_test_words,), dtype=np.int32)
        timer = time()
        for k in range(n_test_words):
            tries[k] = self.test_algorithm_word(algorithm, test_word_inds[k, :], n_words, maxtries)
        avg_time = time() - timer

        return tries, avg_time / n_test_words, setup_time

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

    def __get_algorithm(self, algorithm, algorithm_class):
        if algorithm is None:
            if algorithm_class is None:
                raise TypeError('Missing required argument \'algorithm\' (pos 1)')
            algorithm = algorithm_class(self.words, self.n_bestwords)
        return algorithm

    def test_algorithm_word_input(self, algorithm=None, algorithm_class=None, n_words = 1, maxtries=0):
        algorithm = self.__get_algorithm(algorithm, algorithm_class)
        still = np.full((n_words,), True)
        guess = algorithm.start(n_words)
        answer = np.full((n_words, 5), 2)

        if maxtries == 0:
            generator = itt.count(1)
        else:
            generator = range(1, maxtries)

        for tries in generator:
            print(f'Write \'{guess}\'')
            for k in np.nonzero(still)[0]:
                s = input(f'resposta {tries}, {k+1}: ')
                for j in range(5):
                    answer[k, j] = s[j]
                if np.all(answer[k, :] == 2):
                    still[k] = False

            if not np.any(still):
                break

            guess = algorithm.guess(answer)

if __name__ == '__main__':
    tester = WordleTester('wordle')
    algorithm = solvers.SolutionMatrixSolver(tester.words, tester.n_bestwords, 'wordle')
    tester.test_algorithm_word_input(algorithm)

    print('Bye')
