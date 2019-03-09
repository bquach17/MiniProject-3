import os
import re
import HMM_helper

text = open(os.path.join(os.getcwd(), 'test.txt')).read()
HMM_helper.parse_observations(text)
