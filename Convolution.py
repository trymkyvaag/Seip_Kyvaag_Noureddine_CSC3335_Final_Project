"""
    Authors: Garald Seip
    Spring 2023
    CSC 3335
    
    This file implements a convolution neural network for training tweets with.
    Many elements were inspired by the tutorial here:
    https://realpython.com/python-keras-text-classification
"""
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, GlobalMaxPooling1D, MaxPool1D, Conv1D

from Data_Storage import Data


class Conv_Neur_Net():
    def __init__(self):
        self.data_storage = Data()
        self.label = self.data_storage.PARSED_LABEL
        self.tweet = self.data_storage.PARSED_TWEET
        self.data = self.data_storage.parsed_tweets[[self.tweet, self.label]]
        
        self.tokenize()
        # self.pad()
        
    def tokenize(self):
        self.tweets_train, self.tweets_test, self.label_train, self.label_test = train_test_split(self.data[self.tweet].values,
                                                                                                  self.data[self.label].values,
                                                                                                  test_size = 1/3, 
                                                                                                  random_state = 0)
        
        # Creates and fits the tokenizer based on the number of words in the data set.
        toTok = Tokenizer(num_words = len(self.data_storage.get_unique_words(self.data[self.tweet])))
        toTok.fit(self.tweets_train)
        
        self.tweets_train_tok = toTok.texts_to_sequences(self.tweets_test)
        self.tweets_test_tok = toTok.texts_to_sequences(self.tweets_test)
        
        # Sets the vocab size as determined by the tokenizer. Adds one because zero is reserved.
        self.tok_vocab_size = len(toTok.word_index) + 1
        
    def pad(self):
        """
        I don't think I will ever use this but I figured I should have it just in case.
        This functions pads tweets so they are all the same length.
        """
        # Shortest individual word is one character. 
        # Tweet character cap is 280.
        # 280 / (1 + len(' ')) = 140 
        max_len = 140
        
        self.tweets_train_tok = pad_sequences(self.tweets_train_tok, padding = 'post', maxlen = max_len)
        self.tweets_test_tok  =  pad_sequences(self.tweets_test_tok, padding = 'post', maxlen = max_len)
        
    def createModel(self,
                    hiddenLayerInfo: list[tuple[int, str]] = [(2, 'softplus')], 
                    outputFunc: str = 'softmax', 
                    lossFunction: str = 'categorical_crossentropy',
                    epochs: int = 3,
                    weights: dict[int, float] = None,
                    optimizer: str = 'adam',
                    print_prog: bool = True,
                    print_fin: bool = True):
        model = self.create_model_structure()
        
    def create_model_structure(self, hiddenLayerInfo: list[tuple[int, str]], outputFunc: str):
        """
            This function creates the structure of the model based on the passed parameters.
        """
        # The average english word is 4.7 characters. 4.7 + len(' ') = 5.7
        # The maximum tweet length is 280.
        # ceiling(280 / 5.7) = 50
        output_size = 50
        
        # Creates the model.
        model = Sequential(name = "Multi-Input-Model")
        # Adds the Embedding layer to the model.
        model.add(Embedding(input_dim = self.tok_vocab_size,
                            output_dim = output_size,
                            name = "Embedding-Layer"))
        model.add(Flatten())
        # Used to number the hidden layers.
        i = 1
        # Creates each hidden layer from the passed data.
        for numAndFunc in hiddenLayerInfo:
            # First entry in numAndFunc is the number of nodes, the second is the activation function.
            model.add(Dense(numAndFunc[0], activation = numAndFunc[1], name = "Hidden-Layer-" + str(i) + "-" + numAndFunc[1]))
            i += 1

        # Creates the output layer.
        model.add(Dense(1, activation = outputFunc, name = "Output-Layer-" + outputFunc))

        # Returns the model to caller.
        return model
    
def compileModel(self, model, optimizer, lossFunc):
    """
        This function compiles the model based on the passed data.
    """
    model.compile(
        optimizer = optimizer, 
        loss = lossFunc, 
        metrics = ['Accuracy', 'Precision', 'Recall'],  
        loss_weights = None, 
        weighted_metrics = None, 
        run_eagerly = None, 
        steps_per_execution = None
        )
    
def fitModel(self, model, epochs: int, weights: dict[int, float], print: bool = True):
    """
        This function fits the model based on the passed data.
    """
    # 0 means that no progress bar will be displayed per epoch.
    verbose = 0
    if(print):
        # 1 means a progress bar will be displayed per epoch.
        verbose = 1

    # Fits the model based on the passed parameters.
    model.fit(
        self.tweets_train_tok, 
        self.label_train,
        batch_size = 10, 
        epochs = epochs, 
        verbose = verbose,
        callbacks = None, 
        validation_split = 0.2, 
        shuffle = True, 
        class_weight = weights, 
        sample_weight = None,
        initial_epoch = 0, 
        steps_per_epoch = None, 
        validation_steps = None, 
        validation_batch_size = None, 
        validation_freq = 3, 
        max_queue_size = 10, 
        # Gets the number of CPUs the computer has to use multiprocessing.
        workers = os.cpu_count(), 
        use_multiprocessing = True, 
        )