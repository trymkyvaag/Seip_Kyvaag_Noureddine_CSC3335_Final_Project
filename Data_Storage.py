import re
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None

#Lemmatizer (change word form. Better -> Good, Houses -> House)
from nltk.stem import WordNetLemmatizer
# NLTK Stop words
from nltk.corpus import stopwords as sw


class Data():
    def __init__(self) -> None:
        self.stopwords = sw.words('english')
        self.ltzr = WordNetLemmatizer()
        self.load_concatenated_tweets()
        self.load_parsed_tweets()
        
        # Defines column labels.
        self.PARSED_TWEET = 'Text'
        self.PARSED_LABEL = 'Annotation'
        #self.parsed_tweets.style.set_properties(**{'text-align': 'left'})
        

    def load_concatenated_tweets(self) -> None:
        self.ds_1 = pd.read_csv('data/cyberbullying_tweets.csv')
        self.ds_2 = pd.read_csv('data/FinalBalancedDataset.csv')
        self.ds_3 = pd.read_csv('data/twitter_parsed_dataset.csv')
        self.concatenated_tweets = pd.concat([self.ds_1['tweet_text'], self.ds_2['tweet'], self.ds_3['Text']], ignore_index=True)
        #print(self.ds_2.head())
        # self.processed_Data = self.__process_data__()

    def load_parsed_tweets(self):
        self.parsed_tweets = pd.read_csv('data/twitter_parsed_dataset.csv')

        for idx, tweet in enumerate(self.parsed_tweets['Text']):
            tweet = str(tweet)
            if(tweet.startswith('RT')):
                tweet = tweet[2:]
            self.parsed_tweets['Text'][idx] = self.clean_tweet(tweet)
        

    def clean_tweet(self, tweet: str):
        """
            This function was taken from https://catriscode.com/2021/05/01/tweets-cleaning-with-python/
        Args:
            tweet (str): The tweet to clean.
        Returns:
            str: The cleaned tweet.
        """
        if type(tweet) == float:
            return ""
        temp = tweet.lower()
        
        # Nothing is done with these at the moment, but they could be useful for autotagging tweets.
        hashtags = self.hashtag_extract(tweet)
        
        temp = re.sub("'", "", temp) # to avoid removing contractions in english
        temp = re.sub("@[A-Za-z0-9_]+","", temp)
        temp = re.sub("#[A-Za-z0-9_]+","", temp)
        temp = re.sub(r'http\S+', '', temp)
        temp = re.sub('[()!?]', ' ', temp)
        temp = re.sub('\[.*?\]',' ', temp)
        temp = re.sub("[^a-z0-9]"," ", temp)
        temp = temp.split()
        temp = [w for w in temp if not w in self.stopwords]
        temp = " ".join(word for word in temp)
        return temp

    def lemmatization(self, tweet: str):
        words = tweet.split(' ')

        for i in range(len(words)):
            words[i] = self.ltzr.lemmatize(words[i])

        return ' '.join(words)
    
    def hashtag_extract(self, tweet: str):
        """
        This function grabs all of the hashtags in a tweet and returns them in an array.
        Args:
            tweet (str): The tweet to extract hashtags from
        Returns:
            array[str]: The extracted hashtags are returned.
        """
        words = tweet.split(' ')

        hashtags = []

        for word in words:
            if(word.startswith('#')):
                hashtags.append(word)
        
        return hashtags

    #This needs to make sets make sense in comparison
    def __process_data__(self):
        processed_data = []
        
        for tweet in self.combined_tweets:
            tweet = tweet.lower()
            if 'rt' in tweet:
                tweet = tweet.split('rt')[0]
            processed_data.append(tweet)
            #print(tweet)
        self.processed_data = pd.DataFrame(processed_data)
        pass
    

'''
d = Data()
d.load_parsed_tweets()
print(d.parsed_tweets)
'''