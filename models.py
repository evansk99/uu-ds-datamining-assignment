import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from datetime import datetime

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