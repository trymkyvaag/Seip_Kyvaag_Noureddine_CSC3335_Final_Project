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
        self.LDA(10)
        self.complexity()
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

    def LDA(self, num_topics):
        self.num_topics = num_topics
        self.LDA_model = gensim.models.ldamodel.LdaModel(corpus=self.corpus,
                                           id2word=self.id2word,
                                           num_topics=self.num_topics, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=1000,
                                           passes=100,
                                           alpha='auto',
                                           per_word_topics=True)
        
    def complexity(self):
        print('\nPerplexity: ', self.LDA_model.log_perplexity(self.corpus))  # a measure of how good the model is. lower the better.
        # Compute Coherence Score, using umass score since reccomended by: https://www.baeldung.com/cs/topic-modeling-coherence-score
        self.coherence_model_lda = CoherenceModel(model=self.LDA_model, texts=self.data_lemmatized, dictionary=self.id2word, coherence='u_mass')
        self.coherence_lda = self.coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', self.coherence_lda)


    def test(self):

        #To look at corpus and lemmatized data
        '''
        Testing of diff things
        print(self.corpus[:5])
        print(self.data_lemmatized[:5])
        '''
        # Human readable format of corpus (term-frequency)
        '''
        test = [[(self.id2word[id], freq) for id, freq in cp] for cp in self.corpus[:30]]
        print(test)
        '''
        # Print the Keyword in the 10 topics
        
        pprint(self.LDA_model.print_topics(num_topics=self.num_topics))
        self.doc_lda = self.LDA_model[self.corpus]
        


test = ngram_class(TA())