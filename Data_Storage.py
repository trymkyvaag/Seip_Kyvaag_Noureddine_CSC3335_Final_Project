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
        self.stopwords.extend(['from', 'subject', 're', 'edu', 'use'])
        self.ltzr = WordNetLemmatizer()
        self.load_concatenated_tweets()
        self.load_parsed_tweets()
        #self.parsed_tweets.style.set_properties(**{'text-align': 'left'})
        

    def load_concatenated_tweets(self) -> None:
        self.ds_1 = pd.read_csv('data/cyberbullying_tweets.csv')
        self.ds_2 = pd.read_csv('data/FinalBalancedDataset.csv')
        self.ds_3 = pd.read_csv('data/twitter_parsed_dataset.csv')
        self.ds_4 = pd.read_csv('data/cyberbullying_tweets_2.csv')
        self.ds_5 = pd.read_csv('data/cyberbullying-dataset.csv')
        self.concatenated_tweets = pd.concat([self.ds_1['tweet_text'], 
                                              self.ds_2['tweet'], 
                                              self.ds_3['Text'], 
                                              self.ds_4['headline'],
                                              self.ds_5['TEXT']], 
                                             ignore_index=True)
        self.concatenated_tweets = pd.DataFrame(self.concatenated_tweets).rename(columns={0: 'Text'})

    def load_parsed_tweets(self):
        self.parsed_tweets = pd.read_csv('data/twitter_parsed_dataset.csv')
        
    def clean_parsed(self):
        self.parsed_tweets = self.clean_tweets(self.parsed_tweets, 'Text')
        
    def clean_concatenated(self):
        self.concatenated_tweets = self.clean_tweets(self.concatenated_tweets, 'Text')

    def clean_tweets(self, tweets_to_clean, column_name: str):        
        for idx, tweet in enumerate(tweets_to_clean[column_name]):
            tweet = str(tweet)
            
            tweets_to_clean[column_name][idx] = self.clean_tweet(tweet)
                
        return tweets_to_clean
        

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
        
        # Counts complete retweets as their own tweet.
        if(temp.startswith('RT ')):
            temp = temp[2:]
        
        # If the tweet contains a retweet, splits them up into separate tweets.
        if(temp.__contains__('RT')):
                temp = temp.split('RT')[0]
        
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
        
        # Removes underscores from the tweet.
        if(temp.__contains__('_')):
            temp = temp.replace('_', ' ')
            
        # Handles cases like 'runn ing'.
        if(temp.__contains__(' ing ') or temp.endswith(' ing') or temp.endswith(' ing ')):
            temp = temp.replace(' ing', 'ing')
        
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