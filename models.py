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


def extract_lexical_features(texts):
    features_list = []
    for text in texts:
        features = {}
        sentences = text.split('.')
        words = word_tokenize(text)
        unique_words = set(words)
        features['type_token_ratio'] = len(unique_words) / len(words) if len(words) > 0 else 0
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        features['char_count'] = len(text)
        features['word_count'] = len(words)
        features['sentence_count'] = len([s for s in sentences if s.strip()])
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
    td = pd.DataFrame(vectors.todense())
    td.columns = vectorizer.get_feature_names_out()
    # td['num_tokens'] = df['text_processed'].map(lambda text: len(word_tokenize(text)))
    td['doc'] = df['txt_path'].map(lambda p: p.split('/')[-1])
    # convert to matrix
    dtm = td.to_numpy()[:, :-1]
    return dtm

def LogRegCV(X_train: np.array, y_train: np.array,
           X_test: np.array, y_test: np.array,
           test_fold:int,
           with_bigrams=False, k_folds=5,
        ):
    exec_ts = datetime.now()
    pipeline = make_pipeline(LogisticRegression(
        fit_intercept=True,
        # cv=StratifiedKFold(k_folds, shuffle=True),
        class_weight='balanced',
        random_state=42,
        penalty='l1',
        solver='saga',
        warm_start=True
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
                max_features, test_fold: int,
                with_bigrams=False, k_folds=5
            ):
    print(X_train.shape[1])
    pipeline = make_pipeline(
        MultinomialNB(class_prior=(y_train.sum()/len(y_train), np.count_nonzero(y_train==0)/len(y_train)))
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