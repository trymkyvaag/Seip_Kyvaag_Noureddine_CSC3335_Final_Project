from LDA_analysis import LDA_analysis
from Toakanize import Toakanize
class analyse_tweet():

    def __init__(self) -> None:
        self.lda = LDA_analysis(Toakanize())
        #self.lda.find_best_k(40, 2, 6)
        pass


la = analyse_tweet()