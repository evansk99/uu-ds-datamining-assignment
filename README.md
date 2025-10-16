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
The user needs to choose the desirable parameters at section 4. Experiment Setup. <br>
i)Parameter "feats" can be set to True to feed a set of 9 extra engineered features in addition to the bag of words features to the models. <br>
ii)The function parameter_search can be configured to run based on model_type="decision_tree" or "random_forest", different vectorizer_types=['count', 'tfidf', 'tfidf_noidf'], number of features through feature_sizes=[100, 200, 500, 1000, 2000] and ngram_options=[1, 2] (1 for unigram and 2 for uni+bigrams). The result is an excel export that dictates the cross validation accuracy of each configuration in order to uncover high potential settings for the values of the hyperparameters above.<br>
iii)The funtion tune_and_evaluate can be manually configured with suitable model_type="dt"/"rf" n_features and ngram_range values in order for an exhaustive grid search to provide the best performing model by fine-tuning/testing a set of hyperparameters for each algorithm.
 </p>

##Implementation of Gradient Boosting
Gradient Boosting is executed through run_gradBoost.py. The script reads dataset_df.csv, loads OP Spam v1.4 from DATA_ROOT, vectorizes text with Count or TF-IDF using unigrams or unigrams+bigrams, caps features at 500 or 1,000 with min_df=0.02, trains on folds 1–4, and evaluates on fold 5. Configure options in the Config section, then run python run_gradBoost.py.
i) Choose the text setup via VECTORIZERS=['count','tfidf'], NGRAMS=[(1,1),(1,2)], FEATURE_SIZES=[500,1000], and MIN_DF=0.02.
 ii) randomized_for_combo runs RandomizedSearchCV on GradientBoostingClassifier with five-fold stratified CV, shuffling, accuracy scoring, and random_state=42, drawing 60 configurations that vary: number of trees, learning rate, max depth, subsample, min samples per leaf, min samples to split, and max features.
 iii) The best setting by mean CV accuracy is refit on folds 1–4 and tested on fold 5, exporting test accuracy, precision, recall, F1, the confusion matrix, and per-document predictions.
