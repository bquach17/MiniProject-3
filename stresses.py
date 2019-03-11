import nltk
from functools import lru_cache
from itertools import product as iterprod

try:
    arpabet = nltk.corpus.cmudict.dict()
except LookupError:
    nltk.download('cmudict')
    arpabet = nltk.corpus.cmudict.dict()

def wordbreak(s):
    s = s.lower()
    if s in arpabet:
        return arpabet[s]
    middle = len(s)/2
    partition = sorted(list(range(len(s))), key=lambda x: (x-middle)**2-x)
    for i in partition:
        pre, suf = (s[:i], s[i:])
        if pre in arpabet and wordbreak(suf) is not None:
            return [x+y for x,y in iterprod(arpabet[pre], wordbreak(suf))]
    return None

def checkInt(s):
    '''Check if string contains int'''
    try:
        int(s)
        return True
    except ValueError:
        return False

def stresses(word):
    '''Returns dictionary of
        key: number of syllables
        value: stresses as a string ex. computer: '010' '''
    dicts = {}
    lsts = wordbreak(word)
    st = ''
    for lst in lsts:
        if type(lst) == list:
            s = ''
            for letter in lst:
                if checkInt(letter[-1]):
                    s += letter[-1]
            dicts[len(s)] = s
        else:
            if checkInt(lst[-1]):
                    st += lst[-1]
            dicts[len(st)] = st
    return dicts
