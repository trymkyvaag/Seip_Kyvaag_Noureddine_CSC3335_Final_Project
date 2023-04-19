
'''
Line 5 needs to be ran once 
'''
#import nltk; nltk.download('stopwords')
import re
import numpy as np
import pandas as pd
from pprint import pprint
#https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

#Lemmatizer (change word form. Better -> Good, Houses -> House)
from nltk.stem import WordNetLemmatizer

# Enable logging for gensim 
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

'''
With or without rt???
'''
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

from Data_Storage import Data

class TA:
    def __init__(self) -> None:
        self.data_class = Data()
        self.__toakanize__()
        #print(self.data_class.parsed_tweets['Text'])
        #print(stop_words)
        #print(self.toakanized_data[5])
        pass


    def __toakanize__(self):
        self.toakanized_data = list(self.tweets_to_words())
        

    def train_data(self):
        pass

    #https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
    def tweets_to_words(self):
        for tweet in self.data_class.parsed_tweets['Text']:
                yield(gensim.utils.simple_preprocess(str(tweet)))

        
def test():
    ta = TA()

test()
