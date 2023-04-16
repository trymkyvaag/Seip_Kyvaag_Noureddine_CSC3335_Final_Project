'''
    Authors: Garald Seip
    Spring 2023
    CSC 3335

    This file implements creating n-gram dictionaries for all data in the data set so that they can be classified by bullying type.
'''
from copy import deepcopy
import json
import pandas as pd

class NGram:

    def __init__(self):
        self.BASE_TALLY = {'not_cyberbullying': 0, 'gender': 0, 'religion': 0, 'other_cyberbullying': 0, 'age': 0, 'ethnicity': 0}
        
        self.dataset = pd.read_csv('cyberbullying_tweets.csv')

def incrementValue(self, key: str, initialValue: int = 0):
    if(self.gram_map.get(key) == None):
        self.gram_map.update({key: initialValue})
    else:
        self.gram_map[key] = self.gram_map[key] + 1

def generateNGrams(self,  tweet: str, n_size: int):
    words = tweet.split(' ')

    n_real = n_size - 1

    for i in range(len(words)):
        if(i + n_real < len(words)):
            self.gram_map.update({' '.join(words[i:i+n_size]): deepcopy(self.BASE_TALLY)})

def tally_bullying(self,  tweet: str, b_type: str, n_size: int):
    words = tweet.split(' ')

    n_real = n_size - 1

    for i in range(len(words)):
        if(i + n_real < len(words)):
            incrementValue(self.gram_map[' '.join(words[i:i+n_size])], b_type)

def gen_n_gram_file(self, ngrams: int):
    self.gram_map = {}

    for tweet in self.dataset['tweet_text']:
        generateNGrams(self.gram_map, tweet, ngrams)

    for index, row in self.dataset.iterrows():
        tally_bullying(self.gram_map, row['tweet_text'], row['cyberbullying_type'], ngrams)

    n_gram_file = open(str(ngrams) + '_gram_file.txt', 'w')
    json.dump(self.gram_map, n_gram_file, indent = 4)
    n_gram_file.close()