import numpy as np
import pandas as pd
import os
import matplotlib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer # test it
from scipy.sparse import csr_matrix
from scipy import sparse
from scipy.sparse import hstack
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import precision_score, recall_score

def compute_alpha_values(X_bow: np.array, lexical_X: np.array, y: np.array, base_alpha= 1.0) -> np.array:
    """Given the bag of words representation of the documents 
    computes bow feature specific and class aware smoothing values for multi NB.
    The alpha value is computed for each bow feature based on the feature's class distribution.

    Args:
        X_bow (np.array): bag of words matrix
        lexical_X (np.array): extracted lex features matrix
        y (np.array): train labels

    Returns:
        np.array: An np.array with shape (n_features,) containing the alpha value for each feature
    """
    unique_classes = np.unique(y)
    n_features = X_bow.shape[1]
    bow_alphas = np.ones(n_features)
    for feature_idx in range(n_features):
        feature_counts_by_class = []
        for cls in unique_classes:
            class_mask = (y == cls)
            count_in_class = X_bow[class_mask, feature_idx].sum()
            feature_counts_by_class.append(count_in_class)
        
        feature_counts = np.array(feature_counts_by_class)
        total_count = feature_counts.sum()
        
        if total_count > 0:
            max_ratio = feature_counts.max() / total_count
            if max_ratio > 0.8:  # Highly discriminative word
                bow_alphas[feature_idx] = base_alpha * 0.3
            elif max_ratio < 0.3:  # Appears evenly
                bow_alphas[feature_idx] = base_alpha * 2.0
                
    lexical_alphas = np.array([0.1 for i in range(lexical_X.shape[1])])
    return np.concatenate([bow_alphas,lexical_alphas])

def extract_lexical_features(texts):
    features_list = []
    for text in texts:
        features = {}
        sentences = text.split('.')
        sentence_lengths = [len(sent.split()) for sent in sentences if sent.strip()]
        words = word_tokenize(text)
        unique_words = set(words)
        features['type_token_ratio'] = len(unique_words) / len(words) if len(words) > 0 else 0
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        features['char_count'] = len(text)
        features['word_count'] = len(words)
        features['sentence_count'] = len([s for s in sentences if s.strip()])
        features['avg_sentence_length'] = np.mean(sentence_lengths) if sentence_lengths else 0
        features['punctuation_ratio'] = sum(text.count(p) for p in ['.', ',', '!', '?', ';', ':']) / len(words)
        features['std_sentence_length'] = np.std(sentence_lengths) if sentence_lengths else 0
        #Extra Features Text length and Sentiment (include as extra features)
        analyzer = SentimentIntensityAnalyzer()
        features['sentiment'] = analyzer.polarity_scores(text)['compound']
    
        features_list.append(features)
    return pd.DataFrame(features_list)

def vectorize(df: pd.DataFrame, with_bigrams=False, max_features=150):
    vectorizer = CountVectorizer(
        min_df=0.05, 
        strip_accents='unicode',
        max_features=max_features,
        lowercase=True,
        ngram_range=(1,2) if with_bigrams else (1,1)
    )
    vectors = vectorizer.fit_transform(df.text_processed)
    dtm = vectors.toarray()
    feature_names = vectorizer.get_feature_names_out()
    dtm_df = pd.DataFrame(dtm, index=df.index, columns=feature_names)
    return dtm, dtm_df

def gridSearch(pipeline, params, X, y, k=5):
    gs = GridSearchCV(pipeline, param_grid=params, cv=StratifiedKFold(k, shuffle=True, random_state=42), scoring='f1', verbose=1)
    gs.fit(X,y)
    return gs.best_estimator_
    
def LogRegCV(X_train: np.array, y_train: np.array,
           X_test: np.array, y_test: np.array,
           test_fold:int,
           with_bigrams=False, k_folds=5,
        ):
    exec_ts = datetime.now()
    pipeline = make_pipeline(LogisticRegression(
        fit_intercept=True,
        penalty='l1',
        warm_start=True,
        max_iter=10000
    ))
    classifier = gridSearch(
        pipeline, 
        params={
            'logisticregression__C': [0.1,0.7,1,2],
            'logisticregression__solver': ['liblinear', 'saga']
        },
        X=X_train,y=y_train
    )
    # Define multiple metrics to evaluate
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    cv_results = cross_validate(classifier, X_train, y_train, cv=k_folds, 
                                scoring=scoring, return_train_score=True)
    classifier = classifier.fit(X_train, y_train)
    top5_truthful_terms_idx =  np.argpartition(classifier.steps[0][1].coef_[0], -5)[-5:]
    top5_deceptive_terms_idx = np.argpartition(classifier.steps[0][1].coef_[0], 5)[:5]
    preds = classifier.predict(X_test)
    acc = classifier.score(X_test, y_test)
    avgs = {}
    for score in scoring:
        avgs[f"test_{score}"] = np.mean(cv_results[f"test_{score}"])
        
    fold_accs = {}
    for i,acc in enumerate(cv_results['test_accuracy']):
        fold_accs[f"fold{i+1}"] = round(acc, 3)
    
    accuracies_file = 'plots/logRegr-accuracies-v2.csv'
    logData = {
        'test_accuracy': round(acc, 3),
        "with_bigrams": with_bigrams,
        'execution_time': exec_ts, 
        'num_features': X_train.shape[1],
        'k': k_folds,
        'avg_val_accuracy': round(avgs['test_accuracy'], 3),
        'avg_val_precision': round(avgs['test_precision'], 3),
        'avg_val_recall': round(avgs['test_recall'], 3),
        'avg_val_f1': round(avgs['test_f1'], 3),
    }
    logData.update(fold_accs)
    accuracies = pd.DataFrame([logData])
    if not os.path.exists(accuracies_file):
        accuracies.to_csv(accuracies_file, index=False)
    else:
        accuracies.to_csv(accuracies_file, mode='a', header=False, index=False)
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)  
    filename = f'plots/cm-logRegr-with_bigrams-{exec_ts}.png' if  with_bigrams else f'plots/cm-logRegr-{exec_ts}.png'
    disp.plot().figure_.savefig(filename)
    matplotlib.pyplot.close() 
    return top5_deceptive_terms_idx, top5_truthful_terms_idx


def MultinomialNaiveBayes(X_train: np.array, y_train: np.array,
                X_test: np.array, y_test: np.array,
                alpha: np.array,
                test_fold: int, with_bigrams=False,
                max_features=150, k_folds=5
            ):
    pipeline = make_pipeline(
        MultinomialNB(
            class_prior=(y_train.sum()/len(y_train), np.count_nonzero(y_train==0)/len(y_train)),
            alpha=alpha
        )
    )

    # Define multiple metrics to evaluate
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    # Perform cross-validation
    exec_ts = datetime.now()
    cv_results = cross_validate(pipeline, X_train, y_train, cv=k_folds, 
                                scoring=scoring, return_train_score=True)
    avgs = {}
    for score in scoring:
        avgs[f"test_{score}"] = np.mean(cv_results[f"test_{score}"])
    
    fold_accs = {}
    for i,acc in enumerate(cv_results['train_accuracy']):
        fold_accs[f"fold{i+1}"] = round(acc, 3)
    
    accuracies_file = 'plots/multinomialNB-accuracies-v2.csv'
    classifier = pipeline.fit(X_train, y_train)
    preds = classifier.predict(X_test)
    acc = classifier.score(X_test, y_test)
    logData = {
        'test_accuracy': round(acc, 3),
        "with_bigrams": with_bigrams,
        'execution_time': exec_ts, 
        'num_features': X_train.shape[1],
        'k': k_folds,
        'avg_val_accuracy': round(avgs['test_accuracy'], 3),
        'avg_val_precision': round(avgs['test_precision'], 3),
        'avg_val_recall': round(avgs['test_recall'], 3),
        'avg_val_f1': round(avgs['test_f1'], 3),
    }
    logData.update(fold_accs)
    accuracies = pd.DataFrame([logData])
    if not os.path.exists(accuracies_file):
        accuracies.to_csv(accuracies_file, index=False)
    else:
        accuracies.to_csv(accuracies_file, mode='a', header=False, index=False)
    
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    filename = f'plots/cm-MultiNB-with_bigrams-{exec_ts}.png' if  with_bigrams else f'plots/cm-MultiNB-{exec_ts}.png'
    disp.plot().figure_.savefig(filename)
    matplotlib.pyplot.close() 

def parameter_search(
    df, train_indexes, test_indexes, labels,
    model_type='random_forest',
    feature_sizes=[100, 500, 1000, 2000],
    ngram_options=[1, 2],
    vectorizer_types=['count', 'tfidf', 'tfidf_noidf'],
    extra_features=False,
    extra_train_features=None,
    extra_test_features=None,
    random_state=42, #all random state varibles get the same value 42
    output_excel="Model_Comparisons.xlsx",
    k=5
    ):
    
    results = []

    x_train_text = df.loc[train_indexes, 'text_processed']
    x_test_text = df.loc[test_indexes, 'text_processed']
    y_train = labels[train_indexes]
    y_test = labels[test_indexes]

    for vec_type in vectorizer_types:
        for n_features in feature_sizes:
            for ngram in ngram_options:

                # Vectorizer selection
                if vec_type == 'count':
                    
                    vectorizer = CountVectorizer(
                        min_df=0.05,
                        max_df=0.8,
                        max_features=n_features,
                        lowercase=True,
                        ngram_range=(1, ngram)
                    )
                elif vec_type == 'tfidf':
                    vectorizer = TfidfVectorizer(
                        use_idf=True,
                        min_df=0.05,
                        max_df=0.8,
                        max_features=n_features,
                        lowercase=True,
                        ngram_range=(1, ngram)
                    )
                elif vec_type == 'tfidf_noidf':
                    vectorizer = TfidfVectorizer(
                        use_idf=False,  # TF only (no IDF weighting)
                        min_df=0.05,
                        max_df=0.8,
                        max_features=n_features,
                        lowercase=True,
                        ngram_range=(1, ngram)
                    )

                x_train_vec = vectorizer.fit_transform(x_train_text)
                x_test_vec = vectorizer.transform(x_test_text)

                # Add optional extra features
                if extra_features:
                    extra_train_np = extra_train_features.to_numpy()
                    extra_test_np = extra_test_features.to_numpy()

                    if extra_train_features.ndim == 1:
                        extra_train_np = extra_train_np.reshape(-1, 1)
                        extra_test_np = extra_test_np.reshape(-1, 1)

                    extra_train = csr_matrix(extra_train_np)
                    extra_test = csr_matrix(extra_test_np)

                    x_train_vec = hstack([x_train_vec, extra_train])
                    x_test_vec = hstack([x_test_vec, extra_test])

                # Model setup
                if model_type == 'decision_tree':
                    model = DecisionTreeClassifier(
                        random_state=random_state,
                        max_depth=30,
                        min_samples_split=2,
                        min_samples_leaf=5,
                        criterion='gini',
                        ccp_alpha=0.001
                    )
                    
                elif model_type == 'random_forest':
                    model = RandomForestClassifier(
                        n_estimators=200,
                        max_depth=40,
                        min_samples_split=10,
                        min_samples_leaf=2,
                        max_features='log2',
                        bootstrap=False,
                        random_state=random_state
                    )
                else:
                    raise ValueError("model_type == 'decision_tree' or 'random_forest'")

                # Cross-validation
                skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
                
                fold_accuracies = []

                for train_idx, val_idx in skf.split(x_train_vec, y_train):
                    model.fit(x_train_vec[train_idx], y_train[train_idx])
                    y_val_pred = model.predict(x_train_vec[val_idx])
                    fold_acc = accuracy_score(y_train[val_idx], y_val_pred)
                    fold_accuracies.append(fold_acc)

                mean_acc = np.mean(fold_accuracies)

                # Final test set evaluation
                model.fit(x_train_vec, y_train)
                y_pred = model.predict(x_test_vec)
                test_acc = accuracy_score(y_test, y_pred)

                results.append({
                    'vectorizer': vec_type,
                    'n_features': n_features,
                    'ngram': ngram,
                    'fold_1_acc': fold_accuracies[0],
                    'fold_2_acc': fold_accuracies[1],
                    'fold_3_acc': fold_accuracies[2],
                    'fold_4_acc': fold_accuracies[3],
                    'fold_5_acc': fold_accuracies[4],
                    'cv_mean_acc': mean_acc,
                    'test_acc': test_acc,
                    'k': k
                })

    # Export to Excel
    results_df = pd.DataFrame(results)
    results_df.to_excel(output_excel, index=False)
    print("Results exported in Excel File.")
    
    return results_df

## Fine tune hyperparameters and evaluate best model ##
def tune_and_evaluate(
    x_vec_tr, x_vec_test, labels, train_indexes, test_indexes, 
    model_type="rf", n_features=1000, ngram_range=(1,2), extra_features=False,
    extra_train_features=None, extra_test_features=None, k=5):

    # Change Vectorizer based on the best results of previous step 
    #Decision Trees + Random Forest --> CountVectorizer
    vectorizer = CountVectorizer(
        #use_idf=True,
        #use_idf=False,
        min_df=0.05,
        max_features=n_features,
        lowercase=True,
        ngram_range=ngram_range
    )
    x_train_vec = vectorizer.fit_transform(x_vec_tr)
    x_test_vec = vectorizer.transform(x_vec_test)

    if extra_features==True:
        extra_train_np = extra_train_features.to_numpy()
        extra_test_np = extra_test_features.to_numpy()

        if extra_train_np.ndim == 1:
            extra_train_np = extra_train_np.reshape(-1, 1)
            extra_test_np = extra_test_np.reshape(-1, 1)

        x_train_vec = hstack([x_train_vec, sparse.csr_matrix(extra_train_np)])
        x_test_vec = hstack([x_test_vec, sparse.csr_matrix(extra_test_np)])

    y_train, y_test = labels[train_indexes], labels[test_indexes]

    if model_type == "rf":
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 5, 10],
            'max_features': ['log2'],
            'bootstrap': [True, False],
            'ccp_alpha': [0.0, 0.001, 0.01]
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
        raise ValueError("model_type = rf or dt")

    # CV & Grid Search
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='f1',
        cv=skf,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(x_train_vec, y_train)

    best_model = grid_search.best_estimator_
    best_model_cv_results = run_cv_forests_trees(best_model, x_train_vec, x_test_vec, y_train, y_test, n_features, skf)
    cv_file = f"{model_type}-accuracies.csv"
    if not os.path.exists(cv_file):
        best_model_cv_results.to_csv(cv_file, index=False)
    else:
        best_model_cv_results.to_csv(cv_file, mode='a', header=False, index=False)
    # Evaluate on Test Set
    y_pred = best_model.predict(x_test_vec)

    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)

    # ====== Confusion Matrix ======
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1]).plot()
    plt.title(f"Confusion Matrix - {model_type.upper()}")
    plt.show()

    # Extract Feature Importances
    feature_names = vectorizer.get_feature_names_out().tolist()

    # Append extra feature names if any
    if extra_train_features is not None:
        feature_names.extend(list(extra_train_features.columns))

    importances = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(10)

    # Summary OF Results
    performance_data = {
        'Model_Type': [model_type.upper()],
        'Best_Params': [grid_search.best_params_],
        'CV_Best_F1': [grid_search.best_score_],
        'Test_Accuracy': [test_accuracy],
        'Test_Precision': [test_precision],
        'Test_Recall': [test_recall],
        'Test_F1': [test_f1],
        'n_Features': [n_features],
        'ngram_range': [ngram_range]
    }
    performance_df = pd.DataFrame(performance_data)

    # Save to Excel
    with pd.ExcelWriter("BestModel" + model_type + ".xlsx", engine='openpyxl') as writer:
        performance_df.to_excel(writer, sheet_name='Performance', index=False)
        feature_importance_df.to_excel(writer, sheet_name='Top_Features', index=False)

    # Display Summary
    print(f"\nBest {model_type.upper()} model performance:")
    print(performance_df)
    print("\nTop 10 Most Important Features:")
    print(feature_importance_df)

    return best_model, performance_df, feature_importance_df



def run_cv_forests_trees(best_model, x_train_vec, x_test_vec, y_train, y_test, n_features, skf) -> pd.DataFrame:
    # Cross-validation on training set with the best parameters
    fold_accuracies = []
    
    for train_idx, val_idx in skf.split(x_train_vec, y_train):
        # Fit best model on training fold
        best_model.fit(x_train_vec[train_idx], y_train[train_idx])
        y_val_pred = best_model.predict(x_train_vec[val_idx])
        fold_acc = accuracy_score(y_train[val_idx], y_val_pred)
        fold_accuracies.append(fold_acc)
    
    mean_acc = np.mean(fold_accuracies)
    
    # Evaluate on test set
    best_model.fit(x_train_vec, y_train)
    y_test_pred = best_model.predict(x_test_vec)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    # Save results in a structured way
    results = pd.DataFrame([{
        'n_features': n_features,  # or store other relevant params
        'fold_1_acc': fold_accuracies[0],
        'fold_2_acc': fold_accuracies[1],
        'fold_3_acc': fold_accuracies[2],
        'fold_4_acc': fold_accuracies[3],
        'fold_5_acc': fold_accuracies[4],
        'cv_mean_acc': mean_acc,
        'test_acc': test_acc,
        'k': skf.n_splits
    }])
    return results