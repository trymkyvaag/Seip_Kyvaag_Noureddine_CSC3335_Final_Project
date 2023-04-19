"""
    Authors: Garald Seip
    Spring 2023
    CSC 3335
    
    This file implements a convolution neural network for training tweets with.
    Many elements were inspired by the tutorial here:
    https://realpython.com/python-keras-text-classification
"""
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer

from Data_Storage import Data


class Conv_Neur_Net():
    def __init__(self):
        data = Data()
        self.label = data.PARSED_LABEL
        self.tweet = data.PARSED_TWEET
        self.data = data.parsed_tweets[[self.tweet, self.label]]
        
        self.vectorize()
        self.tokenize()
        
    def vectorize(self):
        self.tweets_train, self.tweets_test, self.label_train, self.label_test = train_test_split(self.data[self.tweet].values,
                                                                                                  self.data[self.label].values,
                                                                                                  test_size = 1/3, 
                                                                                                  random_state = 0)
        # Creates and fits the tokenizer.
        toTok = Tokenizer()
        toTok.fit(self.tweets_train)
        
        self.tweets_train_tok = toTok.texts_to_sequences(self.tweets_test)
        self.tweets_test_tok = toTok.texts_to_sequences(self.tweets_test)
        
    def tokenize(self):
        
        