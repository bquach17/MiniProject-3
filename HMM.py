########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Andrew Kang
# Description:  Set 6 skeleton code
########################################

# You can use this (optional) skeleton code to complete the HMM
# implementation of set 5. Once each part is implemented, you can simply
# execute the related problem scripts (e.g. run 'python 2G.py') to quickly
# see the results from your code.
#
# Some pointers to get you started:
#
#     - Choose your notation carefully and consistently! Readable
#       notation will make all the difference in the time it takes you
#       to implement this class, as well as how difficult it is to debug.
#
#     - Read the documentation in this file! Make sure you know what
#       is expected from each function and what each variable is.
#
#     - Any reference to "the (i, j)^th" element of a matrix T means that
#       you should use T[i][j].
#
#     - Note that in our solution code, no NumPy was used. That is, there
#       are no fancy tricks here, just basic coding. If you understand HMMs
#       to a thorough extent, the rest of this implementation should come
#       naturally. However, if you'd like to use NumPy, feel free to.
#
#     - Take one step at a time! Move onto the next algorithm to implement
#       only if you're absolutely sure that all previous algorithms are
#       correct. We are providing you waypoints for this reason.
#
# To get started, just fill in code where indicated. Best of luck!

import random
import numpy as np
import HMM_helper

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0.
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.

            D:          Number of observations.

            A:          The transition matrix.

            O:          The observation matrix.

            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]


    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''

        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        seqs = [['' for _ in range(self.L)] for _ in range(M + 1)]

        for i, emission in enumerate(x):
            if i == 0:
                # In the beginning, we can simply fill everyting with the current state and the starting
                # probability of transitioning to that particular state
                for j in range(self.L):
                    probs[i + 1][j] = self.A_start[j] * self.O[j][emission]
                    seqs[i + 1][j] = str(j)
            else:
                for next_state in range(self.L):
                    for prev_state in range(self.L):
                        # If we found a higher probability sequence replace the current one
                        if probs[i][prev_state] * self.A[prev_state][next_state] * self.O[next_state][emission] > \
                        probs[i + 1][next_state]:
                            probs[i + 1][next_state] = probs[i][prev_state] * self.A[prev_state][next_state] \
                            * self.O[next_state][emission]
                            seqs[i + 1][next_state] = seqs[i][prev_state] + str(next_state)
        # Return the maximum probability sequence
        max_seq = seqs[M][probs[M].index(max(probs[M]))]
        return max_seq


    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.
        alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        for i, emission in enumerate(x):
            if i == 0:
                # In the beginning, we can simply fill everyting with the current state and the starting
                # probability of transitioning to that particular state
                for j in range(self.L):
                    alphas[i + 1][j] = self.A_start[j] * self.O[j][emission]
            else:
                for next_state in range(self.L):
                    alpha_sum = 0
                    for prev_state in range(self.L):
                        alpha_sum += alphas[i][prev_state] * self.A[prev_state][next_state]
                    alphas[i + 1][next_state] = self.O[next_state][emission] * alpha_sum
            if normalize:
                C = 0
                for j in range(self.L):
                    C += alphas[i + 1][j]
                if C != 0:
                    for j in range(self.L):
                        alphas[i + 1][j] /= C
        return alphas


    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        for i in range(M, 0, -1):
            emission = x[i - 1]
            if i == M:
                for j in range(self.L):
                    betas[i][j] = 1
            else:
                for next_state in range(self.L):
                    for prev_state in range(self.L):
                        betas[i][next_state] += betas[i + 1][prev_state] * \
                        self.A[next_state][prev_state] * self.O[prev_state][x[i]]
            if normalize:
                C = 0
                for j in range(self.L):
                    C += betas[i][j]
                if C != 0:
                    for j in range(self.L):
                        betas[i][j] /= C

        return betas


    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''


        # Calculate each element of A using the M-step formulas.

        init_A = [[0. for _ in range(self.L)] for _ in range(self.L)]

        transition_count = {}
        for i in range(self.L):
            transition_count[i] = 0
        for i in range(len(Y)):
            curr = Y[i]
            for j in range(len(curr) - 1):
                init_A[curr[j]][curr[j + 1]] += 1
                transition_count[curr[j]] += 1
        for i in range(self.L):
            for j in range(self.L):
                init_A[i][j] /= transition_count[i]
        self.A = init_A

        # Calculate each element of O using the M-step formulas.

        init_O = [[0. for _ in range(self.D)] for _ in range(self.L)]

        state_count = {}
        for i in range(self.L):
            state_count[i] = 0
        for i in range(len(Y)):
            currY = Y[i]
            currX = X[i]
            for j in range(len(currX)):
                init_O[currY[j]][currX[j]] += 1
                state_count[currY[j]] += 1
        for i in range(self.L):
            for j in range(self.D):
                init_O[i][j] /= state_count[i]

        self.O = init_O


    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''
        for k in range(N_iters):
            A_num = [[0. for _ in range(self.L)] for _ in range(self.L)]
            A_denom = [0. for _ in range(self.L)]
            O_num = [[0. for _ in range(self.D)] for _ in range(self.L)]
            O_denom = [0. for _ in range(self.L)]
            for x in X:
                alphas = self.forward(x, normalize = True)
                betas = self.backward(x, normalize = True)
                M = len(x)
                for j in range(1, M + 1):
                    P_denom = 0
                    for b in range(self.L):
                        P_denom += alphas[j][b] * betas[j][b]
                    for a in range(self.L):
                        P = alphas[j][a] * betas[j][a]
                        P /= P_denom
                        if j < M:
                            A_denom[a] += P
                        O_num[a][x[j - 1]] += P
                        O_denom[a] += P
                for j in range(1, M):
                    P_denom = 0
                    for a in range(self.L):
                        for b in range(self.L):
                            P_denom += alphas[j][a] * self.A[a][b] * self.O[b][x[j]] * betas[j + 1][b]
                    for a in range(self.L):
                        for b in range(self.L):
                            P = alphas[j][a] * self.A[a][b] * self.O[b][x[j]] * betas[j + 1][b]
                            P /= P_denom
                            A_num[a][b] += P
            for i in range(self.L):
                for j in range(self.L):
                    if A_denom[i] != 0:
                        self.A[i][j] = A_num[i][j] / A_denom[i]
            for i in range(self.L):
                for j in range(self.D):
                    if O_denom[i] != 0:
                        self.O[i][j] = O_num[i][j] / O_denom[i]


    def generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random.

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        emission = []
        states = []

        for i in range(M):
            if i == 0:
                states.append(np.random.choice(range(self.L), p = self.A_start))
                emission.append(np.random.choice(range(self.D), p = self.O[states[0]]))
            else:
                states.append(np.random.choice(range(self.L), p = self.A[states[i - 1]]))
                emission.append(np.random.choice(range(self.D), p = self.O[states[-1]]))

        return emission, states
    
    def generate_stanza(self, num_lines, obs_map_r):
        syb = HMM_helper.get_syllable_dict("data/Syllable_dictionary.txt")
        stanza = ''
        emission = []
        states = []
        j = 0
        
        for i in range(num_lines):
            line = ''
            syllables = 0
            if i == 0:
                states.append(np.random.choice(range(self.L), p = self.A_start))
                emission.append(np.random.choice(range(self.D), p = self.O[states[j]]))
                temp = syb[obs_map_r[emission[j]]]
                for item in temp:
                    if item[0] == 'E':
                        continue
                    else:
                        syllables = int(item)
                        break
                line += obs_map_r[emission[j]].capitalize()
                j += 1
            while syllables != 10:
                states.append(np.random.choice(range(self.L), p = self.A[states[j - 1]]))
                new_emiss = np.random.choice(range(self.D), p = self.O[states[-1]])
                temp = syb[obs_map_r[new_emiss]]
                count = 0
                for item in temp:
                    if item[0] == 'E':
                        continue
                    else:
                        count = int(item)
                        break
                while syllables + count > 10:
                    new_emiss = np.random.choice(range(self.D), p = self.O[states[-1]])
                    temp = syb[obs_map_r[new_emiss]]
                    for item in temp:
                        if item[0] == 'E':
                            continue
                        else:
                            count = int(item)
                            break
                emission.append(new_emiss)
                line += " " + obs_map_r[new_emiss]
                syllables += count
                j += 1
            if num_lines == 4 and i != 3:
                stanza += line + ',\n'
            elif num_lines == 4 and i == 3:
                stanza += line + ':\n'
            elif num_lines == 2 and i == 0:
                stanza += line + ',\n'
            else:
                stanza += line + '.\n'
        return stanza
                        
    
    def generate_poem(self, obs_map_r):
        poem = ''
        for _ in range(3):
            poem += self.generate_stanza(4, obs_map_r)
        poem += self.generate_stanza(2, obs_map_r)
        return poem
    def generate_line(self, obs_map_r, num_syll):
        syb = HMM_helper.get_syllable_dict("data/Syllable_dictionary.txt")
        emission = []
        states = []
        line = ''
        syllables = 0
        j = 0
        states.append(np.random.choice(range(self.L), p = self.A_start))
        emission.append(np.random.choice(range(self.D), p = self.O[states[j]]))
        temp = syb[obs_map_r[emission[j]]]
        for item in temp:
            if item[0] == 'E':
                continue
            else:
                syllables = int(item)
                break
        line += obs_map_r[emission[j]].capitalize()
        j += 1
        while syllables != num_syll:
            states.append(np.random.choice(range(self.L), p = self.A[states[j - 1]]))
            new_emiss = np.random.choice(range(self.D), p = self.O[states[-1]])
            temp = syb[obs_map_r[new_emiss]]
            count = 0
            for item in temp:
                if item[0] == 'E':
                    continue
                else:
                    count = int(item)
                    break
            while syllables + count > num_syll:
                new_emiss = np.random.choice(range(self.D), p = self.O[states[-1]])
                temp = syb[obs_map_r[new_emiss]]
                for item in temp:
                    if item[0] == 'E':
                        continue
                    else:
                        count = int(item)
                        break
            emission.append(new_emiss)
            line += " " + obs_map_r[new_emiss]
            syllables += count
            j += 1
        return line
    
    def generate_haiku(self, obs_map_r):
        syb = HMM_helper.get_syllable_dict("data/Syllable_dictionary.txt")
        stanza = ''
        emission = []
        states = []
        j = 0
        
        for i in range(3):
            line = ''
            syllables = 0
            if i == 0:
                states.append(np.random.choice(range(self.L), p = self.A_start))
                emission.append(np.random.choice(range(self.D), p = self.O[states[j]]))
                temp = syb[obs_map_r[emission[j]]]
                for item in temp:
                    if item[0] == 'E':
                        continue
                    else:
                        syllables = int(item)
                        break
                line += obs_map_r[emission[j]].capitalize()
                j += 1
            if i = 0 or i == 2:
                while syllables != 5:
                    states.append(np.random.choice(range(self.L), p = self.A[states[j - 1]]))
                    new_emiss = np.random.choice(range(self.D), p = self.O[states[-1]])
                    temp = syb[obs_map_r[new_emiss]]
                    count = 0
                    for item in temp:
                        if item[0] == 'E':
                            continue
                        else:
                            count = int(item)
                            break
                    while syllables + count > 10:
                        new_emiss = np.random.choice(range(self.D), p = self.O[states[-1]])
                        temp = syb[obs_map_r[new_emiss]]
                        for item in temp:
                            if item[0] == 'E':
                                continue
                            else:
                                count = int(item)
                                break
                    emission.append(new_emiss)
                    line += " " + obs_map_r[new_emiss]
                    syllables += count
                    j += 1
            else:
                while syllables != 7:
                states.append(np.random.choice(range(self.L), p = self.A[states[j - 1]]))
                new_emiss = np.random.choice(range(self.D), p = self.O[states[-1]])
                temp = syb[obs_map_r[new_emiss]]
                count = 0
                for item in temp:
                    if item[0] == 'E':
                        continue
                    else:
                        count = int(item)
                        break
                while syllables + count > 10:
                    new_emiss = np.random.choice(range(self.D), p = self.O[states[-1]])
                    temp = syb[obs_map_r[new_emiss]]
                    for item in temp:
                        if item[0] == 'E':
                            continue
                        else:
                            count = int(item)
                            break
                emission.append(new_emiss)
                line += " " + obs_map_r[new_emiss]
                syllables += count
                j += 1
            if i == 0 or i == 1:
                stanza += line + ",\n"
            else:
                stanza += line + ".\n"
        return stanza


    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)

    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states, N_iters):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.

        N_iters:    The number of iterations to train on.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM
