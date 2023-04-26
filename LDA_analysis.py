import pickle
from Toakanize import Toakanize
import spacy
import nltk
#nltk.download(), must be ran once
from pprint import pprint
#https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from matplotlib import pyplot as plt

# Enable logging for gensim 
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

class LDA_analysis():
    def __init__(self, load_model) -> None:
        self.nlp = spacy.load("en_core_web_sm")
        if not load_model:
            self.ta_class = Toakanize()
            self.__create_ngrams__()
            self.__clean__()
            self.__create_word_dict__()
            self.__LDA__(8)
            self.__complexity__()
            #self.test()
        else:
            print('\n\n\t-----Using saved model and corpus----\n')
           # self.ta_class = Toakanize()
           # self.__clean__()
            self.num_topics = 8
            self.LDA_model = self.load_model()

        # Define functions for stopwords, bigrams, trigrams and lemmatization

    
    def remove_stopwords(self,texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def make_bigrams(self,texts):
        print('-----Making bi/trigrams-----\n')
        return [self.bigram_mod[doc] for doc in texts]

    def make_trigrams(self,texts):
        return [self.trigram_mod[self.bigram_mod[doc]] for doc in texts]

    def lemmatization(self,texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        print('-----Tagging words-----\n')
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

    def __clean__(self, loaded_model = True):
        print('\n-----Cleaning tweets-----\n')
        if not loaded_model:
   
            # Remove Stop Words
            self.data_words_nostops = self.remove_stopwords(self.ta_class.toakanized_data)

            # Form Bigrams
            self.data_words_bigrams = self.make_bigrams(self.data_words_nostops)

        # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
        # python3 -m spacy download en
        #self.nlp = spacy.load("en_core_web_sm")

        # Do lemmatization keeping only noun, adj, vb, adv
        print('-----Lemmaztion only NN, A, V, AV -----\n')
        self.data_lemmatized = self.lemmatization(self.data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    def __create_word_dict__(self):
        # Create Dictionary
        self.id2word = corpora.Dictionary(self.data_lemmatized)

        # Create Corpus
        texts = self.data_lemmatized

        # Term Document Frequency
        self.corpus = [self.id2word.doc2bow(text) for text in texts]
        pickle.dump(self.corpus, open("corpus.p", "wb")) 

    def __LDA__(self, num_topics):
        self.num_topics = num_topics
        self.LDA_model = gensim.models.ldamodel.LdaModel(corpus=self.corpus,
                                           id2word=self.id2word,
                                           num_topics=self.num_topics, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=2000,
                                           passes=50,
                                           alpha='auto',
                                           per_word_topics=True#,
                                           #minimum_probability=0.03 # THis is not tested yet
                                           )
        self.LDA_model.save('lda.model')
        
    def load_model(self):
       return gensim.models.LdaModel.load('lda.model')
    
    def __complexity__(self):
        print('-----LDA coherence-----\n')
        print('\nPerplexity: ', self.LDA_model.log_perplexity(self.corpus))  # a measure of how good the model is. lower abs the better.
        # Compute Coherence Score, using umass score since reccomended by: https://www.baeldung.com/cs/topic-modeling-coherence-score 
        self.coherence_model_lda = CoherenceModel(model=self.LDA_model, texts=self.data_lemmatized, dictionary=self.id2word, coherence='u_mass')
        self.coherence_lda = self.coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', self.coherence_lda, '\n')
    
    def find_best_k(self, limit, start = None, step = None):
           #Values to adjust
        self.model_rstate = 100
        self.model_update_every = 1
        self.model_chunksize = 1000
        self.model_passes = 50
        print('\n\n-----Model params used: -----\n')
        print(f'\tRandom state seed: {self.model_rstate}')
        print(f'\tUpdate chunk every: {self.model_update_every}')
        print(f'\tModel Chunk size: {self.model_chunksize}')
        print(f'\tNumber passes: {self.model_passes}\n')
        models, values = self.__compute_coherence_values__(limit, start, step)
        print('-----Plotting-----\n')
        x = range(start, limit, 6)
        plt.plot(x, values)
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence score")
        plt.legend(("coherence_values"), loc='best')
        plt.show()

    def __compute_coherence_values__(self, limit, start=2, step=3):
        """
        Compute umass coherence for various number of topics

        Parameters:
        ----------
        dictionary : Gensim dictionary
        corpus : Gensim corpus
        texts : List of input texts
        limit : Max num of topics

        Returns:
        -------
        model_list : List of LDA topic models
        coherence_values : Coherence values corresponding to the LDA model with respective number of topics
        """
        print('-----Finding best K-----:\n')
        coherence_values = []
        model_list = []
        for num_topics in range(start, limit, step):

         
            LDA_model = gensim.models.ldamodel.LdaModel(corpus=self.corpus,
                                           id2word=self.id2word,
                                           num_topics=num_topics, 
                                           random_state=self.model_rstate,
                                           update_every=self.model_update_every,
                                           chunksize=self.model_chunksize,
                                           passes=self.model_passes,
                                           alpha='auto',
                                           per_word_topics=True)
            model_list.append(LDA_model)
            
            coherencemodel = CoherenceModel(model=LDA_model, texts=self.data_lemmatized, dictionary=self.id2word, coherence='u_mass')
            c = coherencemodel.get_coherence()
            coherence_values.append(c)
            print(f'Topics {num_topics}, coherence: {c}')

        print('-----Models and values found-----\n')

        return model_list, coherence_values

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
        # Print the Keyword in the 10 topics in the first model
        
        pprint(self.LDA_model.print_topics(num_topics=self.num_topics))
        #self.doc_lda = self.LDA_model[self.corpus]
        
