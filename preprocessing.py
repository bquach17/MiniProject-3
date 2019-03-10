import os
import re
import HMM_helper
from HMM import unsupervised_HMM

# text = open(os.path.join(os.getcwd(), 'data/shakespeare.txt')).read()
# obs, obs_map = HMM_helper.parse_observations(text)
print(HMM_helper.get_syllable_dict('data/Syllable_dictionary.txt'))
