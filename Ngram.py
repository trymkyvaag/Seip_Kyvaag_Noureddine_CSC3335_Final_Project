from Topic_Analysis import TA
import spacy
import nltk
#nltk.download()
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

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
# from nltk.corpus import brown
# from nltk.book import *
# nltk.help.upenn_tagset('NN')

'''
With or without rt???
'''
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

class ngram_class():
    def __init__(self, ta: TA) -> None:
        self.ta_class = ta
        self.__create_ngrams__()
        self.__clean__()
        self.__create_word_dict__()
        self.test()

        # Define functions for stopwords, bigrams, trigrams and lemmatization
    def remove_stopwords(self,texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def make_bigrams(self,texts):
        return [self.bigram_mod[doc] for doc in texts]

    def make_trigrams(self,texts):
        return [self.trigram_mod[self.bigram_mod[doc]] for doc in texts]

    def lemmatization(self,texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = self.nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    def __create_ngrams__(self):
        self.bigram = gensim.models.Phrases(self.ta_class.toakanized_data, min_count=5, threshold=100) # higher threshold fewer phrases.
        self.trigram = gensim.models.Phrases(self.bigram[self.ta_class.toakanized_data], threshold=100)  
        self.bigram_mod = gensim.models.phrases.Phraser(self.bigram)
        self.trigram_mod = gensim.models.phrases.Phraser(self.trigram)

    def __clean__(self):
        # Remove Stop Words
        self.data_words_nostops = self.remove_stopwords(self.ta_class.toakanized_data)

        # Form Bigrams
        self.data_words_bigrams = self.make_bigrams(self.data_words_nostops)

        # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
        # python3 -m spacy download en
        self.nlp = spacy.load("en_core_web_sm")

        # Do lemmatization keeping only noun, adj, vb, adv
        self.data_lemmatized = self.lemmatization(self.data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])


    def __create_word_dict__(self):
        # Create Dictionary
        self.id2word = corpora.Dictionary(self.data_lemmatized)

        # Create Corpus
        texts = self.data_lemmatized

        # Term Document Frequency
        self.corpus = [self.id2word.doc2bow(text) for text in texts]

        # View
        print(self.corpus[:5])

    def test(self):
        # Remove Stop Words
        #data_words_nostops = self.remove_stopwords(self.ta_class.toakanized_data)
        # Form Bigrams
        #data_words_bigrams = self.make_bigrams(data_words_nostops)
        #print(data_words_bigrams)
        # Do lemmatization keeping only noun, adj, vb, adv
        #data_lemmatized = ngc.lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

        print(self.data_lemmatized[:5])

test = ngram_class(TA())