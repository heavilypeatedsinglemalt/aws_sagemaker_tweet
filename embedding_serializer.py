WORD_DIM = 50
import os
import sys
import pickle
import json
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
"""
Methods for reading in glove embeddings and serializing them, so that they can be stored in s3
The main method reads in glove embeddings, and produces a model.pkl file 
Assumes that there exists a single {}d.model file, where {} is a number representing the word dimension, e.g. 
If this file does not exist, it creates it from a {}d.txt file 
"""
def read_in_or_write_glove():
    """
    Reads in the glove embeddings and stores them in a gensim model
    containing a mapping from words to WORD_DIM dimensional embeddings

    Returns: a gensim model containing a mapping from words to WORD_DIM dimensional embeddings
    """
    model = None
    path_model = os.path.abspath("{}d.model".format(WORD_DIM))
    try:
        model = KeyedVectors.load(path_model)
    except:
        print("could not load word2vec model, this is ok for a first run, or with new WORD_DIM ", sys.exc_info()[0])
        path_glove = os.path.abspath("{}d.txt".format(WORD_DIM))
        print(path_glove)
        glove_file = datapath(path_glove)
        tmp_file = get_tmpfile(path_model)
        _ = glove2word2vec(glove_file, tmp_file)
        model = KeyedVectors.load_word2vec_format(tmp_file)
        model.save(path_model)
    return model
def pickle_object(object, file_dest:str):
    """
    pickles an object and saves serialized representation to disk
    :param object: the object to pickle
    :param file_dest: path to save to
    :return: None
    """
    with open(file_dest,mode = "wb") as file:
        pickle.dump(obj=object, file=file)
def write_model_to_json(model, file_dest:str):
    """
    writes gensim word2vec model with word mappings to json as a python dictionary
    :param model: the gensim word2vec model
    :param file_dest: path to save to
    :return: None
    """

    dictionary = convert_object_to_map(model)
    with open(file_dest, mode="w", encoding="utf8") as file:
        json.dump(obj=dictionary, fp=file)
def convert_object_to_map(model):
    """
    converts gensim model storing word vectors to a simple python dictionary
    :param model: gensim word2vec model
    :return: python dicitonary from word to its embedding representation
    """
    result = {}
    for word in model.vocab:
        result[word] = model[word]
    return result
def load_pickle(file_name:str):
    """
    load pickle from file
    :param file_name: path to load from
    :return: object from pickle
    """
    result = None
    with open(file_name,mode = "rb") as file:
        result = pickle.load(file)
    return result
def load_json(file_name:str):
    """
    loads json serialized object
    :param file_name: path to load from
    :return: object that was serialized
    """
    result = None
    with open(file_name,mode = "r", encoding="utf8") as file:
        result = json.load(fp=file)
    return result
if __name__ == '__main__':
    """
    Reads in glove embeddings, converts them to python dictionary and pickles it 
    """
    model = read_in_or_write_glove()
    model = convert_object_to_map(model)
    model = pickle_object(model,"model.pkl")
    #model = load_pickle("model.pkl")
    #model= load_json("model.json")
    for thing in model.keys:
        print(len(model[thing]))


