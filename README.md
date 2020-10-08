# aws_sagemaker_tweet
Text classification using sagemaker and tensorflow 2.0 serving 

This corresponds to the kaggle challenge: https://www.kaggle.com/c/nlp-getting-started

For the input data to the model refer to the data provided in that challenge. The word dictionary used comes from https://nlp.stanford.edu/projects/glove/ (it's the 50d one). 

The jupyter notebook can be run on AWS Sagemaker (as it also uses the Model API) currently. This is hard to guarantee, because the supported images on AWS are always changing. 

The steps that are taken are as follows: 

## 1. Create an S3 bucket with a pickled python vanilla dictionary in it 
## 2. Upload model and training code to sagemaker vm 
## 3. Train the model 
## 4. Use the artifacts from the trained model that were stored on s3 to deploy a predictor that can classify sentences (disaster or not disaster) 

