import numpy as np
import pandas as pd
import json
import os
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import StandardScaler
from nltk.tokenize import word_tokenize
from sklearn.model_selection import StratifiedKFold
from datetime import datetime

def compute_alpha_values(X_bow: np.array, lexical_X: np.array, y: np.array, base_alpha= 1.0) -> np.array:
    """Given the bag of words representation of the documents 
    computes bow feature specific and class aware smoothing values for multi NB.
    The alpha value is computed for each bow feature based on the feature's class distribution.

    Args:
        X_bow (np.array): bag of words matrix
        lexical_X (np.array): extracted lex features matrix
        y (np.array): train labels

    Returns:
        np.array: An np.array with shape (n_features,) containing the alpha value for each features
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
    return dtm, feature_names

def LogRegCV(X_train: np.array, y_train: np.array,
           X_test: np.array, y_test: np.array,
           test_fold:int,
           with_bigrams=False, k_folds=5,
        ):
    exec_ts = datetime.now()
    pipeline = make_pipeline(LogisticRegression(
        fit_intercept=True,
        # cv=StratifiedKFold(k_folds, shuffle=True),
        C=(2),
        class_weight='balanced',
        random_state=42,
        penalty='l1',
        solver='saga',
        warm_start=True,
        max_iter=50
    ))
    # Define multiple metrics to evaluate
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    cv_results = cross_validate(pipeline, X_train, y_train, cv=k_folds, 
                                scoring=scoring, return_train_score=True)
    classifier = pipeline.fit(X_train, y_train)
    preds = classifier.predict(X_test)
    acc = classifier.score(X_test, y_test)
    avgs = {}
    for score in scoring:
        avgs[f"test_{score}"] = np.mean(cv_results[f"test_{score}"])
    
    accuracies_file = 'plots/logRegr-accuracies.csv'
    accuracies = pd.DataFrame([{
        'test_accuracy': round(acc, 3),
        "with_bigrams": with_bigrams,
        'execution_time': exec_ts, 
        'num_features': X_train.shape[1],
        'k': k_folds,
        'avg_val_accuracy': round(avgs['test_accuracy'], 3),
        'avg_val_precision': round(avgs['test_precision'], 3),
        'avg_val_recall': round(avgs['test_recall'], 3),
        'avg_val_f1': round(avgs['test_f1'], 3),
        'fold': test_fold
    }])
    if not os.path.exists(accuracies_file):
        accuracies.to_csv(accuracies_file)
    else:
        accuracies.to_csv(accuracies_file, mode='a', header=False, index=False)
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)  
    filename = f'plots/cm-logRegr-with_bigrams-{exec_ts}.png' if  with_bigrams else f'plots/cm-logRegr-{exec_ts}.png'
    disp.plot().figure_.savefig(filename)


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
    
    accuracies_file = 'plots/multinomialNB-accuracies.csv'
    # if check_improvement_in_accuracy(avgs['test_accuracy'], with_bigrams, accuracies_file):
    classifier = pipeline.fit(X_train, y_train)
    preds = classifier.predict(X_test)
    acc = classifier.score(X_test, y_test)
    accuracies = pd.DataFrame([{
        'test_accuracy': round(acc, 3),
        "with_bigrams": with_bigrams,
        'execution_time': exec_ts, 
        'num_features': X_train.shape[1],
        'k': k_folds,
        'avg_val_accuracy': round(avgs['test_accuracy'], 3),
        'avg_val_precision': round(avgs['test_precision'], 3),
        'avg_val_recall': round(avgs['test_recall'], 3),
        'avg_val_f1': round(avgs['test_f1'], 3),
        'fold': test_fold
    }])
    if not os.path.exists(accuracies_file):
        accuracies.to_csv(accuracies_file)
    else:
        accuracies.to_csv(accuracies_file, mode='a', header=False, index=False)
    
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    filename = f'plots/cm-MultiNB-with_bigrams-{exec_ts}.png' if  with_bigrams else f'plots/cm-MultiNB-{exec_ts}.png'
    disp.plot().figure_.savefig(filename)
        
def check_improvement_in_accuracy(accuracy: float, with_bigrams: bool, accuracies_file: str):
    try:
        acc_df = pd.read_csv(accuracies_file)
    except Exception:
        return True
    max_acc = max(acc_df[acc_df['with_bigrams'] == with_bigrams].test_accuracy.tolist())
    return round(accuracy, 3) > max_acc