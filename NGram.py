'''
    Authors: Garald Seip
    Spring 2023
    CSC 3335

    This file implements creating n-gram dictionaries for all data in the data set so that they can be classified by bullying type.
'''
from copy import deepcopy
import json
import re
import pandas as pd

class NGram():

    def __init__(self):
        self.BASE_TALLY = {'none': 0, 'sexim': 0, 'racism': 0}
        
        self.dataset = pd.read_csv('twitter_parsed_dataset.csv')

    def incrementValue(self, gram_map: dict, key: str, initialValue: int = 0):
        if(gram_map.get(key) == None):
            gram_map.update({key: initialValue})
        else:
            gram_map[key] = gram_map[key] + 1

    def generateNGrams(self, tweet: str, n_size: int):
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
                self.incrementValue(self.gram_map[' '.join(words[i:i+n_size])], b_type)

    def scrub_punc(self, to_scrub: str):
        to_scrub = to_scrub.lower()
        return re.findall(r'\w+', to_scrub)

    def gen_n_gram_file(self, ngrams: int):
        self.gram_map = {}

        for tweet in self.dataset['tweet_text']:
            self.generateNGrams(' '.join(self.scrub_punc(tweet)), ngrams)

        for index, row in self.dataset.iterrows():
            self.tally_bullying(' '.join(self.scrub_punc(row['Text'])), row['Annotation'], ngrams)

        trigram_file = open(str(ngrams) + '_gram_file.txt', 'w')
        json.dump(self.gram_map, trigram_file, indent = 4)
        trigram_file.close()