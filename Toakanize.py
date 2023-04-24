
'''
Line 5 needs to be ran once 
'''
import gensim
from Data_Storage import Data

class Toakanize:

    '''
        Toakanize class.
        creates a class with the raw data and toakanize it
    '''
    def __init__(self) -> None:
        self.data_class = Data()
        self.data_class.clean_concatenated()
        self.__toakanize__()

    def __toakanize__(self):
        print('-----Toakanizing, vol 1-----')
        self.toakanized_data = list(self.tweets_to_words())

    def tweets_to_words(self):
        for tweet in self.data_class.concatenated_tweets['Text']:
                yield(gensim.utils.simple_preprocess(str(tweet)))

