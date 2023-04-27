from LDA_analysis import LDA_analysis
from Toakanize import Toakanize
from pprint import pprint
import pandas as pd 
import pickle
import tqdm

class analyse_tweet():

    def __init__(self,load_model = True) -> None:
        self.lda = LDA_analysis(load_model)
        #self.print_all_topics()
        self.corpus = pickle.load(open("corpus.p", "rb"))
        '''
        Decided on 8 topics for now    
            self.lda.find_best_k(25, 2, 6)
        '''
        self.topic_names = [('Normal conversation', 0),('Gender based harrasment', 1),('Political conflicts and terrorism', 1)
                            ,('Hate speech and discrimination', 1),('Racism/sexism', 1),('Encouragment', 0),('Religious harrasment and threaths ', 1),('Homophobia/Abelism', 1)] #make array of topic description

        pass

    def analyze_tweet(self, tweet = str, print_readable = True):
        tweet = [[tweet]]
        lem_data = self.lda.lemmatization(tweet)
        new_corp = [self.lda.LDA_model.id2word.doc2bow(text) 
                    for text in lem_data]
        word_prob = self.lda.LDA_model[new_corp]
        topic_str = self.__printable_topics__(word_prob, print_readable)
        return topic_str
    
    def __printable_topics__(self,topics, print_readable = True):
        '''
            Turn topic probabilities to a human readable format 
            or print distrubutions for given tweet
        '''
        if print_readable:   
            print('\nThe probabilities for different topics:\n')
            #Get topic and corresponding prob and print
            '''
                Add functionality for potential topics we do not want
            '''
            pass
        else:
            for topic in topics:
                print('\n-----Print probabilites-----\n')
                pprint(topic)

    def print_all_topics(self):
        '''
            Get just the numerical distrubution of topics
        '''
        pprint(self.lda.LDA_model.show_topics())

la = analyse_tweet(load_model=True)
#la.print_all_topics()
la.analyze_tweet('We need to stop bullying this is not a good thing.', print_readable=False)
passh