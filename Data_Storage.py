import pandas as pd
import numpy as np


class Data():
    def __init__(self) -> None:
        self.ds_1 = pd.read_csv('data/cyberbullying_tweets.csv')
        self.ds_2 = pd.read_csv('data/FinalBalancedDataset.csv')
        self.combined_tweets = pd.concat([self.ds_1['tweet_text'], self.ds_2['tweet']], ignore_index=True)
        print(self.ds_2.head())
        self.__process_data__()

    #This needs to make sets make sense in comparison
    def __process_data__(self):
        processed_data = []
        
        for tweet in self.combined_tweets:
            tweet = tweet.lower()
            if 'rt' in tweet:
                tweet = tweet.split('rt')[0]
            processed_data.append(tweet)
            print(tweet)
        self.processed_data = pd.DataFrame(processed_data)
        pass
    
d = Data()
