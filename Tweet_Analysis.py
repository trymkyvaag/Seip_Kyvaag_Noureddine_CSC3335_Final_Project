from LDA_analysis import LDA_analysis
from Toakanize import Toakanize
class analyse_tweet():

    def __init__(self) -> None:
        self.lda = LDA_analysis(load_model=True)
        '''
        Decided on 8 topics for now    
            self.lda.find_best_k(25, 2, 6)
        '''
        
        pass

    def analyze_tweet(tweet: str):
        pass

la = analyse_tweet()