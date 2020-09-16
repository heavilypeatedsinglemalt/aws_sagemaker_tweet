import argparse
import os
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import tweet_model
import tweet_utils
from tensorflow.keras.optimizers import SGD
WORD_DIM = 50
def _get_args():
    """
    Retrieves arguments passed to the computing instance, some must be taken from environment variables, see:

    https://sagemaker.readthedocs.io/en/stable/using_tf.html

    Returns:

        arguments passed to container from sagemaker notebook instance
    """
    parser = argparse.ArgumentParser()
    # Training Parameters
    parser.add_argument('--batch_size', type=int, default=13, metavar='N',
                        help='input batch size for training (default: 13)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # Model Parameters
    parser.add_argument('--embedding_dim', type=int, default=32, metavar='N',
                        help='size of the word embeddings (default: 32)')
    parser.add_argument('--hidden_dim', type=int, default=100, metavar='N',
                        help='size of the hidden dimension (default: 100)')
    parser.add_argument('--pad_length', type=int, default=30, metavar='N',
                        help='length to pad the sentences to ')
    # SageMaker Parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data_dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    args = parser.parse_args()
    return args
def _get_train_data(training_dir):
    """
    Read in input data from training directory passed from sagemaker:

    https://sagemaker.readthedocs.io/en/stable/using_tf.html

    Returns:

        Pandas dataframe where last column is target
    """
    print("read train data from input dir")
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"))
    train_data.describe()
    return train_data
def _get_word_dict(training_dir):
    """
    Read in word dictionary from training directory passed from sagemaker:

    https://sagemaker.readthedocs.io/en/stable/using_tf.html

    The word dictionary is a pickled dictionary

    Returns:

        map from word to 50 dimensional embedding
    """
    result = None
    with open(os.path.join(training_dir,"model.pkl"),mode = "rb") as file:
        result = pickle.load(file)
    print("word dict:")
    print(result)
    return result
if __name__ == '__main__':
    """
    loads arguments passed in through sagemaker 
    loads training data fed in through sage maker 
    creates LSTM model with hyperparameters passed in through arguments
    saves model in tensorflow serving format to model_path = os.path.join(local_model_path, "1")
    where local_model_path is the available environment variable under os.environ['SM_MODEL_DIR'] taken from sagemaker
    """
    args = _get_args()
    training_data = _get_train_data(args.data_dir)
    word_dict = _get_word_dict(args.data_dir)
    pad_length = args.pad_length
    x_train, y_train = tweet_utils.preprocess_data_tf(training_data, word_dict,
                                                      pad_length=pad_length,
                                                      is_test_data=False,
                                                      word_dim = WORD_DIM)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    hyperparameters = {"unit_multiplier": 1, "num_lstm_stacks": 2, "pad_length": pad_length, "word_dim": WORD_DIM}
    model = tweet_model.LSTMClassifier(word_dict, hyperparameters)
    model.compile(optimizer=SGD(), loss='binary_crossentropy',metrics=['accuracy'])
    model.fit(x=x_train,y=y_train, epochs = args.epochs, verbose = False,batch_size =args.batch_size)
    local_model_path = os.environ['SM_MODEL_DIR']
    predict = model.predict.get_concrete_function(tf.TensorSpec(shape=[1], dtype=tf.string))
    signatures = {"serving_default": predict,"predict": predict}
    model_path = os.path.join(local_model_path, "1")
    print("model_path", model_path)
    tf.saved_model.save(model,model_path, signatures=signatures)




