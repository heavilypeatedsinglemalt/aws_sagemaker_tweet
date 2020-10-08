import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, LSTM, BatchNormalization, Conv1D
from tensorflow.keras.layers import MaxPool1D
"""
A simple LSTM network for binary text classification 

Can be served and is intended to be used in such a way to be used with tensorflow serving such that it accepts
single sentences and produces a classification

It stores word embeddings that have been provided to it in its constructor in a tensorflow servable form, so that 
no artifacts need to be retrieved during serving, or through a separate endpoint
"""
class LSTMClassifier(tf.keras.Model):
    def _build_network(self):
        """
        Builds a stacked LSTM network

        Parameters:

            unit_multiplier: a number to multiply the number of hidden units by

            num_lstm_stacks: the number of lstm stacks

        Returns:

            Keras LSTM stacked network
        """
        unit_multiplier = self.hyperparameters["unit_multiplier"]
        num_lstm_stacks = self.hyperparameters["num_lstm_stacks"]
        pad_length = self.hyperparameters["pad_length"]
        word_dim = self.hyperparameters["word_dim"]
        BASE_NUM_FILTERS = 100  # the number of filters to use for the conv1d layer for the lst mnetwork
        BASE_NUM_UNITS = 50  # the number of hidden units
        KERNEL_SIZE = 10  # kernel size for the 1D convolution in the network
        print("model built")
        model = tf.keras.Sequential()
        model.add(Conv1D(filters=BASE_NUM_FILTERS * unit_multiplier, kernel_size=KERNEL_SIZE, activation='relu',
                         input_shape=(pad_length, word_dim)))
        model.add(MaxPool1D())
        for i in range(1, num_lstm_stacks):
            model.add(
                LSTM(units=BASE_NUM_UNITS * unit_multiplier, return_sequences=True,dropout=.50))
            model.add(BatchNormalization())
        model.add(LSTM(units=BASE_NUM_UNITS * unit_multiplier, return_sequences=False))
        model.add(BatchNormalization())
        model.add(Dense(units=1, activation="sigmoid"))
        model.run_eagerly = True
        model.build()
        self.model = model
    def _build_lookup_tables(self):
        """
        Takes arguments from the hyperparameters and word dict and builds up a vectorized representation that can be
        saved for serving. modifies self.word_to_index, self.padding, self.word_vectors, self.pad_length

        word_to_index is a lookup table with keys based on words in the word dictionary,
         and values the index in the dict, so looking up a word gives an integer
        padding is for sentences smaller than pad_length
        word_vectors are the embeddings taken from the dictionary passed to the constructor
        """
        pad_length = self.hyperparameters["pad_length"]
        word_dim = self.hyperparameters["word_dim"]
        word_dict = self.word_dict
        input_keys = [key for key in word_dict.keys()]
        table_keys = tf.constant(input_keys, dtype=tf.string)
        input_values = [value for value in word_dict.values()]
        index_range = tf.range(start=0, limit=table_keys.get_shape()[0])
        init = tf.lookup.KeyValueTensorInitializer(keys=table_keys, values=index_range)
        self.word_to_index = tf.lookup.StaticHashTable(initializer=init, default_value=0)
        self.padding = tf.constant([[0,0],[0,pad_length]])
        self.word_vectors = tf.constant(input_values, dtype=tf.double, shape=(len(input_values),word_dim))
        self.pad_length = pad_length
    def _convert_sentences_to_tensors(self,text_tensor):
        """
        Creates a ragged tensor, representing tokens of a string that has been processed with a regular expression:
        [^A-Za-z ] (remove special/numeric chars), [^ ]*http[^ ]* (remove links)
        :param text_tensor: string tensor
        :return: ragged tensor of strings
        """
        removal_non_char = tf.strings.regex_replace(input=text_tensor, pattern="[^A-Za-z ]", rewrite="")
        remove_links = tf.strings.regex_replace(input=removal_non_char, pattern="[^ ]*http[^ ]*", rewrite="")
        stripped_spaces = tf.strings.strip(remove_links)
        replace_space_with_comma = tf.strings.regex_replace(input=stripped_spaces, pattern=" +", rewrite=",")
        split_by_comma = tf.strings.split(input=replace_space_with_comma, sep=",")
        return split_by_comma
    def _convert_tensors_to_features(self, processed_text_tensor):
        """
        Takes a tokenized string ragged and converts it to an embedding taken from the word_dict passed to constructor
        :param processed_text_tensor: tokenized string represented by a ragged tensor
        :return:tensor of doubles reprensenting embeddings for every word in a sentence
        """
        mapped_words_to_indexes = tf.ragged.map_flat_values(lambda value: self.word_to_index.lookup(value),
                                                            processed_text_tensor)
        mapped_words_to_indexes = mapped_words_to_indexes.to_tensor(default_value=0)
        padded_index_sequences = tf.pad(mapped_words_to_indexes, self.padding)
        sliced_index_sequences = tf.slice(padded_index_sequences, [0, 0],
                                          [tf.shape(padded_index_sequences)[0], self.pad_length])
        embedding_sequences = tf.nn.embedding_lookup(self.word_vectors, sliced_index_sequences)
        return embedding_sequences
    def __init__(self,word_dict,hyperparameters):
        """
        :param word_dict: word dictionary, represented by python dict where strings are mapped to numpy arrays or 2d
        :param hyperparameters: hyperparameters passed to training instance, including padding_length, word_dim
        """
        super(LSTMClassifier, self).__init__()
        self.word_dict = word_dict
        self.hyperparameters = hyperparameters
        self._build_network()
        self._build_lookup_tables()
    def call(self, inputs):
        """
        applies the network to an embedding vector and produces a tensor of values representing probabilities
        (for binary classification)
        :param inputs: sentences of word embeddings
        :return: tensor of probabilities, where each element is the probability corresponding to a sentence
        """
        return self.model(inputs)
    @tf.function(input_signature=[tf.TensorSpec(shape=[1],dtype=tf.string)])
    def predict(self, stuff):
        """
        function to be used with tensor serving api, takes one dimensional array of strings representing a sentence
        returns classification of sentence (disaster or non-disaster)
        :param stuff: array of strings, converted to tensor of strings representing a sentence, or multiple sentences
        :return: tensor of probabilties, where each probability corresponds to classification of sentence
        """
        processed_text_tensor = self._convert_sentences_to_tensors(stuff)
        feature_vectors = self._convert_tensors_to_features(processed_text_tensor)
        return self(feature_vectors)

