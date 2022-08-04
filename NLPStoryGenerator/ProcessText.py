from collections import Counter
import numpy as np
import re, string

punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation + "/\\")))
num_regex = re.compile('[{}]'.format(re.escape("0123456789")))
def clean(in_string):
    in_string.replace("'", "")
    strip = num_regex.sub('',punc_regex.sub(' ', in_string))
    strip.replace("â\x80\x94", "")
    strip.replace("â\x80\x99", "")
    strip.replace("â\x80", " ")
    strip.replace("  ", " ")
    strip.replace("  ", " ")
    strip.replace("  ", " ")
    strip.replace("  ", " ")
    return strip.lower()
def tokenize(in_string):
    strip = punc_regex.sub('', in_string)
    return strip.lower().split()

def create_bag_of_words(text, vocabulary=10000) -> Counter:
    return Counter(tokenize(text)).most_common(vocabulary)

def InverseFrequency(frequencies: Counter, doc_count: int):
    N = doc_count
    # nt = np.array(nt, dtype=float)
    return {words : np.log10(N/counts) for words, counts in frequencies.most_common()}