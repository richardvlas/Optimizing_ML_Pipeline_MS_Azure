# Optimizing an ML Pipeline in Azure

## Overview

In this project, I had the opportunity to create and optimize an Azure ML pipeline. I have been provided a custom-coded model—a standard Scikit-learn Logistic Regression--which  hyperparameters I optimized using HyperDrive. I also used AutoML to build and optimize a model on the same dataset, so that I can compare the results of the two methods.

You can see the main steps that I have taken in the diagram below:

![image](creating-and-optimizing-an-ml-pipeline.png)

## Summary
The dataset used in this project is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to assess if the product (bank term deposit) would be ('yes') or not ('no') subscribed which is what we are predicting in this project.

So the aim of the project is to classify if a potential prospect would subcribe to the bank's term deposit. 

The best model found using AutoML experiment run has achieved classification prediction accuracy of 91.68% with VotingEnsemble model whereas the Logistic regression model with optimized hyperparameters from HyperDrive has achieved accuracy of 91.697%. 

## Scikit-learn Pipeline
The pipeline consists of a custom-coded Scikit-learn model logistic regression model stored in train.py script and a Hyperdrive run sweeping over model paramters. The following steps are part of the pipeline:
- Data cleaning and converting categorical data into one hot encoded data
- Splitting data into train and test sets
- Setting logistic regression parameters: 
    - --C - Inverse of regularization strenght 
    - --max_iter - Maximum number of iterations convergence
- Azure Cloud resources configuration
- Creating a HyperDrive configuration using the estimator, hyperparameter sampler, and policy
- Retrieve the best run and save the model from that run

**RandomParameterSampling**
Defines random sampling over a hyperparameter search space. In this sampling algorithm, parameter values are chosen from a set of discrete values or a distribution over a continuous range. This has an advantage against GridSearch method that runs all combinations of parameters and requires large amount of time to run.

For the Inverse of regularization strenght parameter I have chosen uniform distribution with min=0.001 and max=1.0 
For the Maximum number of iterations convergence I inputed a range of values (5, 25, 50, 100, 150)

**BanditPolicy Class**
Defines an early termination policy based on slack criteria, and a frequency and delay interval for evaluation. This greatly helps to ensure if model with given parameters is not performing well, it is turned down instead of running it for any longer.  

The best model given by HyperDrive resulted in training accuracy of 91.697%. The hyperparameters of the model are as follows:
- --C = 0.7104
- --max_iter = 25


## AutoML
Automated machine learning, also referred to as automated ML or AutoML, is the process of automating the time-consuming, iterative tasks of machine learning model development. 

The steps to implement AutoML are the following:
- Create TabularDataset using TabularDatasetFactory
- Data cleaning and converting categorical data into one hot encoded data
- Concatenate features with label column
- Splitting data into train and test sets
- Creating a dataset in Azure ML
- Setting up parameters for AutoML to find the best model and paramters

The best model found by AutoML was a VotingEnsemble having training accuracy of 91.68%. 

A VotingEnsemble is an ensemble machine learning model that combines the predictions from multiple other models. The VotingEnsemble class used by AutoMLConfig object defines an ensemble created from previous AutoML iterations that implements soft voting, which uses weighted averages. 

The AutoML generated an ensemble of 5 ML models consisting of XGBoostClassifier and LightGBM classifier.

A sample of hyperparameters for one XGBoostClassifier in the ensemble is shown below:

- 'base_score': 0.5
- 'booster': 'gbtree'
- 'colsample_bylevel': 1
- 'colsample_bynode': 1
- 'colsample_bytree': 0.9
- 'eta': 0.3
- 'gamma': 0
- 'learning_rate': 0.1
- 'max_delta_step': 0
- 'max_depth': 10
- 'max_leaves': 15
- 'min_child_weight': 1
- 'missing': nan
- 'n_estimators': 25
- 'n_jobs': 1
- 'nthread': None
- 'objective': 'reg:logistic'
- 'random_state': 0
- 'reg_alpha': 0
- 'reg_lambda': 0.5208333333333334
- 'scale_pos_weight': 1
- 'seed': None
- 'silent': None
- 'subsample': 0.6
- 'tree_method': 'auto'
- 'verbose': -10
- 'verbosity': 0


## Pipeline comparison
Both models performed similarly with respect to the training accuracy with HyperDrive giving the accuracy of 91.697% and AutoML with accuracy of 91.68%. The advantage of AutoML is that it allows to test a large variety of ML algoritms when compared to a single algorithm being tuned by HyperDrive. This is a real advantage as we can easily evaluate many different algorithms and make sure we selected the right one. 

## Future work
Some areas of improvement are the following:
- Data balancing. The dataset is highly unbalanced resulting in a biased ML model 
- Change the range of hyperparameters to extend the search space 
- Use of Neural Networks even thought this could require large dataset. A test to see how a simple NN architecture would perform is a worth to investigate

