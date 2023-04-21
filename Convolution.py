"""
    Authors: Garald Seip
    Spring 2023
    CSC 3335
    
    This file implements a convolution neural network for training tweets with.
    Many elements were inspired by the tutorial here:
    https://realpython.com/python-keras-text-classification
"""
import os
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, GlobalMaxPooling1D, MaxPool1D, Conv1D

from Data_Storage import Data


class Conv_Neur_Net():
    def __init__(self):
        print('{:50s}'.format('Loading and cleaning data.'), end = '\r')
        self.data_storage = Data()
        
        self.label = self.data_storage.PARSED_LABEL
        self.tweet = self.data_storage.PARSED_TWEET
        self.data = self.data_storage.parsed_tweets[[self.tweet, self.label]]
        
        # The location of the GloVe file.
        self.GLOVE_LOC = 'temp'
        
        # Shortest individual word is one character. 
        # Tweet character cap is 280.
        # 280 / (1 + len(' ')) = 140 
        self.max_len = 140
        
        print('{:50s}'.format('Tokenizing data.'), end = '\r')
        self.tokenize()
        self.pad()  
        
        # print('{:50s}'.format('Doing stupid conversion part 1.'), end = '\r')
        # temp = np.zeros((len(self.tweets_train_tok), self.max_len, 1))
        
        # for o in range(len(self.tweets_train_tok)):
        #     for i in range(self.max_len):
        #         temp[o][i][0] = self.tweets_train_tok[o][i]
        # self.tweets_train_tok = temp
        
        # print('{:50s}'.format('Doing stupid conversion part 2.'), end = '\r')
        # temp = np.zeros((len(self.tweets_test_tok), self.max_len, 1))
        
        # for o in range(len(self.tweets_test_tok)):
        #     for i in range(self.max_len):
        #         temp[o][i][0] = self.tweets_test_tok[o][i]
        # self.tweets_test_tok = temp
        
    def tokenize(self):
        self.tweets_train, self.tweets_test, self.label_train, self.label_test = train_test_split(self.data[self.tweet].values,
                                                                                                  self.data[self.label].values,
                                                                                                  # Uses 20% of the total data for testing.
                                                                                                  test_size = 1/5, 
                                                                                                  random_state = 0)
        
        # Creates and fits the tokenizer based on the number of words in the data set.
        print(len(self.data_storage.get_unique_words(self.data[self.tweet])))
        self.toTok = Tokenizer(num_words = len(self.data_storage.get_unique_words(self.data[self.tweet])))
        self.toTok.fit_on_texts(self.tweets_train)
        
        self.tweets_train_tok = self.toTok.texts_to_sequences(self.tweets_train)
        self.tweets_test_tok = self.toTok.texts_to_sequences(self.tweets_test)
        
        # Sets the vocab size as determined by the tokenizer. Adds one because zero is reserved.
        self.tok_vocab_size = len(self.toTok.word_index) + 1
        
    def pad(self):
        """
        I don't think I will ever use this but I figured I should have it just in case.
        This functions pads tweets so they are all the same length.
        """
        
        self.tweets_train_tok = pad_sequences(self.tweets_train_tok, padding = 'post', maxlen = self.max_len)
        self.tweets_test_tok  =  pad_sequences(self.tweets_test_tok, padding = 'post', maxlen = self.max_len)
        
    def create_model(self,
                    conv_layer_info: tuple[int, int, str] = (32, 3, 'relu'),
                    hidden_layer_info: list[tuple[int, str]] = [(2, 'softplus')], 
                    output_func: str = 'softmax',
                    use_pretrained: bool = False, 
                    loss_function: str = 'binary_crossentropy',
                    epochs: int = 3,
                    weights: dict[int, float] = None,
                    optimizer: str = 'adam',
                    print_prog: bool = True,
                    print_fin: bool = True,):
        print('{:50s}'.format('Creating model structure.'), end = '\r')
        self.create_model_structure(conv_layer_info, hidden_layer_info, output_func, use_pretrained)
        print('{:50s}'.format('Compiling model.'), end = '\r')
        self.compile_model(optimizer, loss_function)
        print('{:50s}'.format('Fitting model.'), end = '\r')
        self.fit_model(epochs, weights, print_prog)
        
    def create_model_structure(self, 
                               conv_layer_info: tuple[int, int, str], 
                               hidden_layer_info: list[tuple[int, str]], 
                               output_func: str, 
                               use_pretrained: bool):
        """
            This function creates the structure of the model based on the passed parameters.
        """
        # The average english word is 4.7 characters. 4.7 + len(' ') = 5.7
        # The maximum tweet length is 280.
        # ceiling(280 / 5.7) = 50
        output_size = 50
        
        # Creates the model.
        self.model = Sequential(name = "Multi-Input-Model")
        # Adds the Embedding layer to the model.
        if(not use_pretrained):
            self.model.add(Embedding(input_dim = self.tok_vocab_size,
                                     output_dim = output_size,
                                     name = "Embedding-Layer",
                                     input_length = self.max_len))
        else:
            # self.pad()
            emb_matrix = self.load_word_embeddings(self.tok_vocab_size, output_size)
            self.model.add(Embedding(input_dim = self.tok_vocab_size,
                                     output_dim = output_size,
                                     weights = [emb_matrix],
                                     input_length = self.max_len,
                                     trainable = True))
        
        # Adds the convolution layer to the model.
        # self.model.add(Conv1D(filters = conv_layer_info[0],
        #                       # It seems the general rule with convolution is to go deeper, not wider.
        #                       kernel_size = conv_layer_info[1],
        #                       activation = conv_layer_info[2]))
        
        # Adds global max pooling to the model.
        self.model.add(GlobalMaxPooling1D())
        # Used to number the hidden layers.
        i = 1
        # Creates each hidden layer from the passed data.
        for numAndFunc in hidden_layer_info:
            # First entry in numAndFunc is the number of nodes, the second is the activation function.
            self.model.add(Dense(numAndFunc[0], activation = numAndFunc[1], name = "Hidden-Layer-" + str(i) + "-" + numAndFunc[1]))
            i += 1

        # Creates the output layer.
        self.model.add(Dense(1, activation = output_func, name = "Output-Layer-" + output_func))
    
    def compile_model(self, optimizer, loss_func):
        """
            This function compiles the model based on the passed data.
        """
        self.model.compile(
            optimizer = optimizer, 
            loss = loss_func, 
            metrics = ['Accuracy', 'Precision', 'Recall'],  
            loss_weights = None, 
            weighted_metrics = None, 
            run_eagerly = None, 
            steps_per_execution = None
            )
        
    def fit_model(self, epochs: int, weights: dict[int, float], print_prog: bool = True):
        """
            This function fits the model based on the passed data.
        """
        # 0 means that no progress bar will be displayed per epoch.
        verbose = 0
        if(print_prog):
            # 1 means a progress bar will be displayed per epoch.
            verbose = 1
            
        print(self.tweets_train[1337])
        print(self.tweets_train_tok[1337])
        print(self.label_train[1337])

        # Fits the model based on the passed parameters.
        self.model.fit(
            self.tweets_train_tok, 
            self.label_train,
            batch_size = 10, 
            epochs = epochs, 
            verbose = verbose,
            callbacks = None, 
            # Uses 20% of the total data for validation.
            # validation_split = 1/4, 
            validation_data = (self.tweets_test_tok, self.label_test),
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
        
    def load_word_embeddings(self, num_tokens: int, embedding_dim: int):
        """This function loads GloVe's pretrained word embeddings for use in the text classification.
        I wanted to use Google's word2vec embeddings, but I could not find an example of doing so online.
        Code taken and modified from the Keras example: https://keras.io/examples/nlp/pretrained_word_embeddings/

        Args:
            num_tokens (int): The number of tokens from tokenization.
            embedding_dim (int): The output of the embedding dimension.

        Returns:
            _type_: A matrix of embeddings is returned.
        """
        embeddings_index = {}
        with open(self.GLOVE_LOC) as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                embeddings_index[word] = coefs
                
        hits = 0
        misses = 0

        # Prepare embedding matrix
        embedding_matrix = np.zeros((num_tokens, embedding_dim))
        for word, i in self.toTok.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                # This includes the representation for "padding" and "OOV"
                embedding_matrix[i] = embedding_vector
                hits += 1
            else:
                misses += 1
                
        print("Percentage of words found in pre-trained embeddings: %d%", hits / misses)
        return embedding_matrix
    
    def printModel(self):
        """
            This function prints out details about the model.
        """
        # Prints the model summary.
        self.printModelSummary(self.model)
        # Prints the model weights.
        self.printModelWeights(self.model)
        # Prints the model evaluated on the training and testing data.
        self.printModelEvaluations(self.label_train, self.label_test, self.training_output, self.testing_output)
        print()

    def printModelSummary(self, model):
        """
            This function prints the model summary.
        """
        print('\n----------------------- Model Summary -----------------------')
        model.summary()

    def printModelWeights(self):
        """
            This function prints the model weights.
        """
        print('\n--------------------- Weights and Biases --------------------')
        for layer in self.model.layers:
            print(layer.name)
            currData = layer.get_weights()
            print("  --Weights: " + str(currData[0]))
            print("  --Biases: ", str(currData[1]))

    def printModelEvaluations(self, training_output, testing_output):
        """
            This function prints how accurate the models predictions are on the training and testing data.
        """
        print('\n---------------- Evaluation on Training Data ----------------')
        print(classification_report(self.label_train, training_output))

        print('\n------------------ Evaluation on Test Data ------------------')
        print(classification_report(self.label_test, testing_output))