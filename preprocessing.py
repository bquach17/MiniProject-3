import os
import re
import HMM_helper

text = open(os.path.join(os.getcwd(), 'data/shakespeare.txt')).read()
obs, obs_map = HMM_helper.parse_observations(text)
