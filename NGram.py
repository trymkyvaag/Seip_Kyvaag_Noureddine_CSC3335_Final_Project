'''
    Authors: Garald Seip
    Spring 2023
    CSC 3335

    This file implements creating n-gram dictionaries for all data in the data set so that they can be classified by bullying type.
'''
from copy import deepcopy
import json
import pandas as pd

BASE_TALLY = {'not_cyberbullying': 0, 'gender': 0, 'religion': 0, 'other_cyberbullying': 0, 'age': 0, 'ethnicity': 0}

def incrementValue(theDict: dict, key: str, initialValue: int = 0):
    if(theDict.get(key) == None):
        theDict.update({key: initialValue})
    else:
        theDict[key] = theDict[key] + 1

def generateNGrams(gram_map: dict, tweet: str, n_size: int):
    words = tweet.split(' ')

    n_real = n_size - 1

    for i in range(len(words)):
        if(i + n_real < len(words)):
            gram_map.update({' '.join(words[i:i+n_size]): deepcopy(BASE_TALLY)})

def tally_bullying(gram_map: dict, tweet: str, b_type: str, n_size: int):
    words = tweet.split(' ')

    n_real = n_size - 1

    for i in range(len(words)):
        if(i + n_real < len(words)):
            incrementValue(gram_map[' '.join(words[i:i+n_size])], b_type)

dataset = pd.read_csv('cyberbullying_tweets.csv')
gram_map = {}

for tweet in dataset['tweet_text']:
    generateNGrams(gram_map, tweet, 3)

for index, row in dataset.iterrows():
    tally_bullying(gram_map, row['tweet_text'], row['cyberbullying_type'], 3)

trigram_file = open('trigram_file.txt', 'w')
json.dump(gram_map, trigram_file, indent = 4)
trigram_file.close()