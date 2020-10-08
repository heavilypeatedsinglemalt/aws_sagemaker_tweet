# aws_sagemaker_tweet
Text classification using sagemaker and tensorflow 2.0 serving 

This corresponds to the kaggle challenge: https://www.kaggle.com/c/nlp-getting-started

For the input data to the model refer to the data provided in that challenge. The word dictionary used comes from https://nlp.stanford.edu/projects/glove/ (it's the 50d one). 

The jupyter notebook can be run on AWS Sagemaker (as it also uses the Model API) currently. This is hard to guarantee, because the supported images on AWS are always changing. 

The most interesting thing about this is that the model stores its own preprocessing methods as tensorflow operations, so an additional step for preprocessing is not required for serving. 

The steps that are taken are as follows: 

## 1. Create an S3 bucket with a pickled python vanilla dictionary in it 
## 2. Upload model and training code to sagemaker vm 
## 3. Train the model 
## 4. Use the artifacts from the trained model that were stored on s3 to deploy a predictor that can classify sentences (disaster or not disaster) 

About the files contained herein: 

**embedding_serializer.py**: this is a tool used to convert the word dictionary from https://nlp.stanford.edu/projects/glove/ to a readable dictionary, that is then serialized (pickled) 

**sagemaker_notebook.ipynb**: the notebook that was uploaded to the sagemaker vm, contains documentation describing what each step does 

**train.py**: the training code that is also to be uploaded to the sagemaker vm, and used to train the model and upload its artifacts to s3 as a tensorflow serving model, it also passed arguments to the model that are stored in its serving model implicitly through the persistence of tensorflow operations in the model 

**tweet_model.py**: contains the model that is a stacked LSTM network. The most interesting thing about this is that it converts a dictionary to a tensorflow operation that can be used in the tensorflow serving deployment. The purpose for doing this is that the 2.0 vanilla version of tensorflow serving only understands tensorflow operations and not python code, so we can't do any real preprocessing using established python operations without incurring additional cost. However, in this model it actually contains its own preprocessing mechanism, which saves the hastle of knowing what kind of artifacts (in this case, embedding dictionaries) the model relies on. 

**tweet_utils.py**: contains methods used in training to undersample the data to reduce mode variance due to bias (in this case underrepresentation of a class), as well as text preprocessing steps that are necessary for the raw data read into pandas from the csv file from https://www.kaggle.com/c/nlp-getting-started

