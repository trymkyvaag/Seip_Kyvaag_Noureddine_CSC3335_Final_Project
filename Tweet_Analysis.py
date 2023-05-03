from LDA_analysis import LDA_analysis
from Toakanize import Toakanize
from pprint import pprint
import pandas as pd 
import pickle
import gensim
from Response_Array import Response_generator

class analyse_tweet():

    def __init__(self,load_model = True) -> None:
        self.lda = LDA_analysis(load_model)
        self.response_genrator = Response_generator()
        #self.print_all_topics()
        self.corpus = pickle.load(open("corpus.p", "rb"))
        '''
            Decided on 8 topics for now    
            self.lda.find_best_k(25, 2, 6)
        '''
        '''
            Named topic for LDA model as of Apr 27th 11:28 am
            0 is an ok topic, 1 is likely offensice
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
        # print(f'The tweet {tweet} has the current breakdown\n')
        topic_str = self.__printable_topics__(word_prob, print_readable)
        return topic_str
    
    def __printable_topics__(self,word_prob, print_readable = True):
        '''
            Turn topic probabilities to a human readable format 
            or print distrubutions for given tweet
        '''
        to_return = ''
        
        if print_readable:   
            #Get topic and corresponding prob and print
            '''
                Topic_tuple is ("name of topic", 1/0 (offensice or not)) a
                Prob is decimal probability 
            '''
            
            topic_name_tuple, prob = self.__get_dominant_topic__(word_prob)
            to_return += self.response_genrator.get_response(topic_name_tuple[1]) + '\n'
            to_return += f'This tweet looks to fall under the category: [{topic_name_tuple[0]}].'

            if topic_name_tuple[1]:
                to_return += f'This falls under this category with {round(prob,2)}% '
            else:
                to_return += f'It falls under the category with {round(prob,2)}% probability. '

        else:
            to_return += '\n-----Print probabilites-----\n'
            pprint(word_prob[0])
            
        # print(to_return)
        return to_return

    def __get_dominant_topic__(self, word_prob = gensim.interfaces.TransformedCorpus):
        most_probable_topic = 0
        topic_list = [list(element) for element in word_prob[0][0]]
        topic_list[5][1] = topic_list[5][1]/2
        for entry in topic_list:
            if entry[1] > most_probable_topic:
                most_probable_topic = entry[1]
                e = entry
        return self.topic_names[e[0]], most_probable_topic*100
        
    def print_all_topics(self):
        '''
            Get just the numerical distrubution of topics
        '''
        pprint(self.lda.LDA_model.show_topics())

# la = analyse_tweet(load_model=True)
# la.analyze_tweet('I hate all canadians', print_readable=True)
# la.analyze_tweet('I hate all canadians', print_readable=False)

# pass