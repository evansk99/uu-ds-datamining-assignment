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

# My imports
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer # test it
from scipy.sparse import csr_matrix
from sklearn.metrics import precision_score, recall_score


#Unchanged
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

#Unchanged
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

#Unchanged
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

#Unused
def check_improvement_in_accuracy(accuracy: float, with_bigrams: bool, accuracies_file: str):
    try:
        acc_df = pd.read_csv(accuracies_file)
    except Exception:
        return True
    prev_accuracies = acc_df[acc_df['with_bigrams'] == with_bigrams].accuracy.tolist()
    return all(round(accuracy, 3) > prev_acc for prev_acc in prev_accuracies)


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
    output_excel="Model_Comparisons.xlsx"):
    
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
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
                
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
                    'test_acc': test_acc
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
    extra_train_features=None, extra_test_features=None):

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
            'n_estimators': [50, 100, 200, 500],
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
        raise ValueError("model_type = rf or dt")

    # CV & Grid Search
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
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
