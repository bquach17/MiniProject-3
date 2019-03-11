import os
import re
from HMM_helper import sample_sentence, parse_observations
from HMM import unsupervised_HMM

text = open(os.path.join(os.getcwd(), 'data/shakespeare.txt')).read()
obs, obs_map = parse_observations(text)

hmm4 = unsupervised_HMM(obs, 4, 100)
print('Sample Sentence 4 Hidden States:\n====================')
print(sample_sentence(hmm4, obs_map, n_words=25))

hmm8 = unsupervised_HMM(obs, 10, 100)
print('\nSample Sentence 10 Hidden States:\n====================')
print(sample_sentence(hmm8, obs_map, n_words=25))

hmm16 = unsupervised_HMM(obs, 16, 100)
print('\nSample Sentence 16 Hidden States:\n====================')
print(sample_sentence(hmm16, obs_map, n_words=25))
