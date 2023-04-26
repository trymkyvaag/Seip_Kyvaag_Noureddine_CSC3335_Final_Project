from LDA_analysis import LDA_analysis
from Toakanize import Toakanize
from pprint import pprint
import pandas as pd 
import pickle


class analyse_tweet():

    def __init__(self) -> None:
        self.lda = LDA_analysis(load_model=True)
        self.corpus = pickle.load(open("corpus.p", "rb"))
        self.test = self.analyze_dataset()
        print(self.test.head())

        #self.print_topics()
        '''
        Decided on 8 topics for now    
            self.lda.find_best_k(25, 2, 6)
        '''

        pass

    def analyze_tweet(self, tweet = str):
        tweet = [[tweet]]
        lem_data = self.lda.lemmatization(tweet)
        new_corp = [self.lda.LDA_model.id2word.doc2bow(text) 
                    for text in lem_data]
        word_prob = self.lda.LDA_model[new_corp]
        return word_prob
    
    def __printable_topics__(self, topics):
        '''
            Turn topic probabilities to a human readable format
        '''
        pass
        '''
        for idx, topics in enumerate(self.lda.LDA_model.show_topics()):
            wp = self.lda.LDA_model.show_topic(idx)
            topic_keywords = ", ".join([word for word, prop in wp])
            sent_topics_df = sent_topics_df.append(pd.Series([int(idx), topic_keywords]), ignore_index=True)
            pass
        
        sent_topics_df = pd.DataFrame()
        # Get main topic in each document
        for i, row in enumerate(self.lda.LDA_model[self.corpus]):
            row = sorted(row, key=lambda x: (x[0]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = self.lda.LDA_model.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        # Add original text to the end of the output
        contents = pd.Series(self.ta_class.data)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        return(sent_topics_df)
        '''
        

    def print_topics(self):
        '''
            Maybe have this print to the gui?
        '''
        pprint(self.lda.LDA_model.show_topics())

la = analyse_tweet()