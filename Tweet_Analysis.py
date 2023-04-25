from LDA_analysis import LDA_analysis
from Toakanize import Toakanize
from pprint import pprint
import pandas as pd 
import pickle


class analyse_tweet():

    def __init__(self) -> None:
        self.lda = LDA_analysis(load_model=True)
        self.corpus = pickle.load(open("corpus.p", "rb"))
        self.analyze_tweet('str')
        self.print_topics()
        '''
        Decided on 8 topics for now    
            self.lda.find_best_k(25, 2, 6)
        '''

        pass

    def analyze_tweet(self, tweet: str):
        for idx, topics in enumerate(self.lda.LDA_model.show_topics()):
            wp = self.lda.LDA_model.show_topic(idx)
            topic_keywords = ", ".join([word for word, prop in wp])
            sent_topics_df = sent_topics_df.append(pd.Series([int(idx), topic_keywords]), ignore_index=True)
            pass
        pass

    def print_topics(self):
        '''
            Maybe have this print to the gui?
        '''
        pprint(self.lda.LDA_model.show_topics())

la = analyse_tweet()