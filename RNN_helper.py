import re
import numpy as np

def parse_observations(text):
    # Convert text to dataset.
    lines = [line.split() for line in re.split('\:|\;|\.', text) if line.split()]

    obs_counter = 0
    obs = ""
    
    for line in lines:

        try:
            int(line[0])
            line = line[1:]
        except ValueError:
            line = line
        for word in line:
            word = re.sub(r'[^\w]', '', word).lower()
            obs += word
            obs += " "
    return obs


