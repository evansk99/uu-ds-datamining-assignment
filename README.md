# uu-ds-datamining-assignment

## Install Dependencies
`$ pip install -r requirements.txt`

## Implementation of Logistic Regression and Naive Bayes
 <p>
    Naive Bayes is executed in `run_MultiNB_logReg.py`. Its only tunable parameter `alpha` is an array computed from the training data beforehand. The scripts runs multiple iterations with different configurations of number of splits and the max number of features BoW should consider ranging in the values {5,10} and {150,200,250,300,350,400,450,500,550,600}.
 </p>
 <p>
    Logistic Regression is executed in the same file the same way, with the addition of grid search cv to find the best configuration for `C` and `solver`
 </p>

## Implementation of Classification Tree and Random Forest
<p>
Both classification tree and random forest algorithms can be executed through `run_decTree_randFor.py` by setting the desired model_type. 
The user needs to choose the desirable parameters at section 4. Experiment Setup. 
i)Parameter "feats" can be set to True to feed a set of 9 extra engineered features in addition to the bag of words features to the models. 
ii)The function parameter_search can be configured to run based on model_type="decision_tree" or "random_forest", different vectorizer_types=['count', 'tfidf', 'tfidf_noidf'], number of features through feature_sizes=[100, 200, 500, 1000, 2000] and ngram_options=[1, 2] (1 for unigram and 2 for uni+bigrams). The result is an excel export that dictates the cross validation accuracy of each configuration in order to uncover high potential settings for the values of the hyperparameters above.
iii)The funtion tune_and_evaluate can be manually configured with suitable model_type="dt"/"rf" n_features and ngram_range values in order for an exhaustive grid search to provide the best performing model by fine-tuning/testing a set of hyperparameters for each algorithm.
 </p>
