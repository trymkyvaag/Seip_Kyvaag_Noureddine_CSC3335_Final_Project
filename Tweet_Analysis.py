from LDA_analysis import LDA_analysis
from Toakanize import Toakanize
from pprint import pprint


class analyse_tweet():

    def __init__(self) -> None:
        self.lda = LDA_analysis(load_model=True)
        self.print_topics()
        '''
        Decided on 8 topics for now    
            self.lda.find_best_k(25, 2, 6)
        '''

        pass

    def analyze_tweet(tweet: str):
        pass

    def print_topics(self):
        pprint(self.lda.LDA_model.show_topics())

la = analyse_tweet()