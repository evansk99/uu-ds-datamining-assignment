import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from datetime import datetime

from scipy import sparse
from scipy.sparse import hstack

## after collective experiment function
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer # test it

def vectorize(df: pd.DataFrame, with_bigrams=False, max_features=150):
    vectorizer = CountVectorizer(
        min_df=0.05, #removes features that do not appear in at least 5% of the data
        strip_accents='unicode',
        max_features=max_features,
        lowercase=True,
        ngram_range=(1,2) if with_bigrams else (1,1)
    )
    vectors = vectorizer.fit_transform(df.text_processed) #sparse dataframe with processed text
    td = pd.DataFrame(vectors.todense()) #make it dense
    td.columns = vectorizer.get_feature_names_out() #get features names out
    # td['num_tokens'] = df['text_processed'].map(lambda text: len(word_tokenize(text)))
    td['doc'] = df['txt_path'].map(lambda p: p.split('/')[-1]) #add column with filename ath the end
    # convert to matrix
    dtm = td.to_numpy()[:, :-1]
    return dtm

def LogRegCV(X_train: np.array, y_train: np.array,
           X_test: np.array, y_test: np.array,
           with_bigrams=False
        ):
    exec_ts = datetime.now()
    classifier = LogisticRegressionCV(
        fit_intercept=True,
        cv=StratifiedKFold(20, shuffle=True),
        class_weight='balanced',
        random_state=42,
        penalty='l1',
        solver='saga',
        # refit=True,
        # intercept_scaling=1,
        # verbose=True
    ).fit(X_train,y_train)
    
    preds = classifier.predict(X_test)
    probs = classifier.predict_proba(X_test)
    logits = classifier.predict_log_proba(X_test)
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    N = X_train.shape[1]
    acc = (y_test == preds).sum() / N
    accuracies_file = 'plots/logRegr-accuracies.csv'
    if check_improvement_in_accuracy(acc, with_bigrams):
        accuracies = pd.DataFrame([{'accuracy': round(acc, 3), "with_bigrams": with_bigrams, 'execution_time': exec_ts, 'k': classifier.cv.n_splits}])
        if not os.path.exists(accuracies_file):
            accuracies.to_csv(accuracies_file)
        else:
            accuracies.to_csv(accuracies_file, mode='a', header=False, index=False)
        filename = f'plots/cm-logRegr-with_bigrams-{exec_ts}.png' if  with_bigrams else f'plots/cm-logRegr-{exec_ts}.png'
        disp.plot().figure_.savefig(filename)

def MultinomialNaiveBayes(X_train: np.array, y_train: np.array,
                X_test: np.array, y_test: np.array,
                with_bigrams=False
            ):
    classifier = MultinomialNB().fit(X=X_train, y=y_train)
    preds = classifier.predict(X_test)
    acc = classifier.score(X_test, y_test)
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    exec_ts = datetime.now()
    accuracies_file = 'plots/multinomialNB-accuracies.csv'
    if check_improvement_in_accuracy(acc, with_bigrams, accuracies_file):
        accuracies = pd.DataFrame([{'accuracy': round(acc, 3), "with_bigrams": with_bigrams, 'execution_time': exec_ts, 'num_features': X_train.shape[-1]}])
        if not os.path.exists(accuracies_file):
            accuracies.to_csv(accuracies_file)
        else:
            accuracies.to_csv(accuracies_file, mode='a', header=False, index=False)
        filename = f'plots/cm-multinomialNB-with_bigrams-{exec_ts}.png' if  with_bigrams else f'plots/cm-multinomialNB-{exec_ts}.png'
        disp.plot().figure_.savefig(filename)

def check_improvement_in_accuracy(accuracy: float, with_bigrams: bool, accuracies_file: str):
    try:
        acc_df = pd.read_csv(accuracies_file)
    except Exception:
        return True
    prev_accuracies = acc_df[acc_df['with_bigrams'] == with_bigrams].accuracy.tolist()
    return all(round(accuracy, 3) > prev_acc for prev_acc in prev_accuracies)


def run_tree_experiment(df, train_indexes, test_indexes, labels, extra_train_features=None, 
                        extra_test_features=None,
                        model_type='random_forest', feature_sizes=[50, 100, 200, 500, 1000, 2000],
                        ngram_options=[1,2], random_state=42, extra_features=False):
    """
    Run experiments with Decision Tree or Random Forest classifiers.

    Parameters:
    - df: DataFrame with preprocessed text
    - train_indexes, test_indexes: lists of indices
    - labels: numpy array of labels
    - model_type: 'decision_tree' or 'random_forest'
    - feature_sizes: list of max_features for vectorization
    - ngram_options: list of 1 (unigrams) or 2 (unigrams+bigrams)
    - extra_features: list of extra column names to append (optional)
    
    Returns:
    - results_df: CV F1 results
    - test_results_df: Test F1 and accuracy
    - last_model: trained model on full training data
    """

    results = []
    test_results = []

    # Split text and labels
    x_train_text = df.loc[train_indexes, 'text_processed']
    x_test_text = df.loc[test_indexes, 'text_processed']
    y_train, y_test = labels[train_indexes], labels[test_indexes]

    for n_features in feature_sizes:
        for ngram in ngram_options:

            # Vectorization
            vectorizer = TfidfVectorizer(
                use_idf=True, 
                min_df=0.05,
                max_df=0.8,
                max_features=n_features,
                lowercase=True,
                ngram_range=(1, ngram)
            )
            x_train_vec = vectorizer.fit_transform(x_train_text)
            x_test_vec = vectorizer.transform(x_test_text)
            
            # Optionally add extra features
            if extra_features:
                # Convert pandas Series/DataFrame to numpy arrays
                extra_train_np = extra_train_features.to_numpy()
                extra_test_np = extra_test_features.to_numpy()

                # Ensure 2D shape
                if extra_train_features.ndim == 1:
                    extra_train_np = extra_train_np.reshape(-1, 1)
                    extra_test_np = extra_test_np.reshape(-1, 1)

                # Convert to sparse matrix
                extra_train = sparse.csr_matrix(extra_train_np)
                extra_test = sparse.csr_matrix(extra_test_np)

                # Stack with text features
                x_train_vec = hstack([x_train_vec, extra_train])
                x_test_vec = hstack([x_test_vec, extra_test])

            # Model selection
            if model_type == 'decision_tree':
                model = DecisionTreeClassifier(
                    random_state=random_state,
                    max_depth=30,
                    min_samples_split=2,
                    min_samples_leaf=5,
                    criterion='gini',
                    ccp_alpha=0.01
                )

            elif model_type == 'random_forest':
                model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=40,
                    min_samples_split=10,
                    min_samples_leaf=2,
                    max_features='log2',
                    bootstrap=False,
                    random_state=42
                )
            else:
                raise ValueError("model_type must be 'decision_tree' or 'random_forest'")
                           
            # Cross-validated F1
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_f1 = np.mean(cross_val_score(model, x_train_vec, y_train, cv=skf, scoring='f1'))

            # Fit full model and predict
            model.fit(x_train_vec, y_train)
            y_pred = model.predict(x_test_vec)
            test_f1 = f1_score(y_test, y_pred)
            test_acc = accuracy_score(y_test, y_pred)

            # Save results
            results.append({'n_features': n_features, 'ngram': ngram, 'cv_f1': cv_f1})
            test_results.append({'n_features': n_features, 'ngram': ngram,
                                 'test_f1': test_f1, 'test_acc': test_acc})

    results_df = pd.DataFrame(results)
    test_results_df = pd.DataFrame(test_results)

    return results_df, test_results_df, model

def tune_and_evaluate(x_vec_tr, x_vec_test, labels, train_indexes, test_indexes, 
                      model_type="rf", n_features=500, ngram_range=(1,1)):
    """
    Fine-tune hyperparameters for Random Forest or Decision Tree,
    train best model, and evaluate on test set.

    Parameters
    ----------
    x_vec_tr : Series (train text data)
    x_vec_test : Series (test text data)
    labels : array-like (full labels array)
    train_indexes : list of indices for training set
    test_indexes : list of indices for test set
    model_type : str ("rf" for Random Forest, "dt" for Decision Tree)
    n_features : int, number of features for vectorizer
    ngram_range : tuple, ngram range for vectorizer
    """

    #Vectorize text
    vectorizer = TfidfVectorizer( ########
        use_idf=True, 
        min_df=0.05,
        max_features=n_features,
        lowercase=True,
        ngram_range=ngram_range
    )
    x_train_vec = vectorizer.fit_transform(x_vec_tr)
    x_test_vec = vectorizer.transform(x_vec_test)

    #Labels
    y_train, y_test = labels[train_indexes], labels[test_indexes]

    #Choose model and parameter grid
    if model_type == "rf":
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 500],
            'max_depth': [10, 20, 30, 50, None],
            'min_samples_split': [2, 5, 10, 15, 20],
            'min_samples_leaf': [1, 2, 5, 10, 15],
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [True, False]
        }
    elif model_type == "dt":
        model = DecisionTreeClassifier(random_state=42)
        param_grid = {
            'max_depth': [5, 10, 15, 20, 30, None],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 5, 10],
            'criterion': ['gini', 'entropy'],
            'ccp_alpha': [0.0, 0.001, 0.005, 0.01, 0.1]
        }
    else:
        raise ValueError("model_type must be 'rf' (Random Forest) or 'dt' (Decision Tree)")

    # Cross-validation & Grid Search
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='f1',
        cv=skf,
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(x_train_vec, y_train)

    # Best model
    best_model = grid_search.best_estimator_

    #Evaluate on test set
    y_pred = best_model.predict(x_test_vec)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    
    ##print(f"Best parameters for {model_type.upper()}:", grid_search.best_params_)
    ##print("Best CV F1-score:", grid_search.best_score_)
    ##print("Test Accuracy:", test_accuracy)
    ##print("Test F1-score:", test_f1)

    #Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    disp.plot()
    plt.show()

    return best_model, grid_search.best_params_,grid_search.best_score_, test_accuracy, test_f1