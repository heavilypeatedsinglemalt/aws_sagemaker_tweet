import numpy as np
import pandas as pd
import tensorflow as tf
def sub_sample(df_input):
    """
    Performs undersampling for imbalanced samples where positive findings are underrepresented, a cheap hack for having too little data

    Returns: A random sampling of df_input as a dataframe where target == 0 and target == 1 are balanced

    """
    count_negative = len(df_input[df_input["target"] == 0])
    print("Number of negative samples", count_negative)
    count_positive = len(df_input[df_input["target"] == 1])
    print("Number of positive samples", count_positive)
    sample_fraction = count_positive / count_negative
    print("Resampling negative as fraction", sample_fraction)
    sample_zero = df_input[df_input["target"] == 0].sample(frac=sample_fraction, random_state=20)
    sample_one = df_input[df_input["target"] == 1]
    result_frame = pd.concat([sample_zero, sample_one], axis=0)
    result_frame = result_frame.sample(frac=1.0, random_state=30).reset_index(drop=True)
    return result_frame
def preprocess_data_tf(df_input, word_vectors, pad_length, is_test_data=False,  word_dim=50):
    """
    Subsamples data if not test data, processes input text using gensim, returns tuple of text mapped to vector of word embeddings and target values

    Parameters:

        df_input: input dataframe

        word_vectors: map from word to WORD_DIM dimensional embedding

        pad_length: maximum length of tweet (to pad/truncate to)

        is_test_data: if for submission or not, deactivates subsampling for test data used for submission and returns ids instead of targets

    Returns:

        returns tuple of text mapped to vector of word embeddings and target values or ids if for test_data
    """
    if is_test_data is False:
        df_input = sub_sample(df_input)
    text_input = tf.constant(np.array(df_input['text']),dtype=tf.string)
    feature_vectors = preprocess_data_tf_op(text_input,word_vectors,pad_length,word_dim)
    if is_test_data is False:
        return (feature_vectors, df_input["target"])
    else:
        return (feature_vectors, df_input["id"])
def preprocess_data_tf_op(text_tensor, word_vectors, pad_length,  word_dim=50):
    """
    Subsamples data if not test data, processes input text using gensim, returns tuple of text mapped to vector of word embeddings and target values

    Parameters:

        df_input: input dataframe

        word_vectors: map from word to WORD_DIM dimensional embedding

        pad_length: maximum length of tweet (to pad/truncate to)

        is_test_data: if for submission or not, deactivates subsampling for test data used for submission and returns ids instead of targets

    Returns:

        returns tuple of text mapped to vector of word embeddings and target values or ids if for test_data
    """
    removal_non_char = tf.strings.regex_replace(input=text_tensor, pattern="[^A-Za-z ]", rewrite="")
    remove_links = tf.strings.regex_replace(input=removal_non_char, pattern="[^ ]*http[^ ]*", rewrite = "")
    stripped_spaces = tf.strings.strip(remove_links)
    replace_space_with_comma = tf.strings.regex_replace(input=stripped_spaces, pattern=" +", rewrite = ",")
    split_by_comma = tf.strings.split(input=replace_space_with_comma,sep=",")
    input_keys = [ key for key in word_vectors.keys()]
    table_keys = tf.constant(input_keys,dtype = tf.string)
    input_values = [value for value in word_vectors.values()]
    word_dim = len(input_values[0])
    table_values = tf.constant(input_values, dtype = tf.double, shape=(len(input_values),word_dim))
    index_range = tf.range(start=0,limit=table_keys.get_shape()[0])
    init = tf.lookup.KeyValueTensorInitializer(keys=table_keys,values = index_range)
    table = tf.lookup.StaticHashTable(initializer=init,default_value =0)
    mapped_words_to_indexes = tf.ragged.map_flat_values(lambda value : table.lookup(value),split_by_comma)
    padding_stuff = tf.constant([[0,0],[0,pad_length]])
    mapped_words_to_indexes = mapped_words_to_indexes.to_tensor(default_value=0)
    padded_index_sequences = tf.pad(mapped_words_to_indexes,padding_stuff)
    sliced_index_sequences = tf.slice(padded_index_sequences,[0,0],[tf.shape(padded_index_sequences)[0],pad_length])
    embedding_sequences = tf.nn.embedding_lookup(table_values,sliced_index_sequences)
    return embedding_sequences
def preprocess_data_raw(df_input, is_test_data=False,  word_dim=50):
    """
    Subsamples data if not test data, processes input text using gensim, returns tuple of text mapped to vector of word embeddings and target values

    Parameters:

        df_input: input dataframe

        word_vectors: map from word to WORD_DIM dimensional embedding

        pad_length: maximum length of tweet (to pad/truncate to)

        is_test_data: if for submission or not, deactivates subsampling for test data used for submission and returns ids instead of targets

    Returns:

        returns tuple of text mapped to vector of word embeddings and target values or ids if for test_data
    """
    if is_test_data is False:
        df_input = sub_sample(df_input)
    feature_vectors = list()
    for text in df_input["text"]:
        feature_vectors.append(text)
    if is_test_data is False:
        return (feature_vectors, df_input["target"])
    else:
        return (feature_vectors, df_input["id"])