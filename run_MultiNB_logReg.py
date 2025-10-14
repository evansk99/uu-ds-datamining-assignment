import os
import pandas as pd
import string
import nltk
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle
from models import LogRegCV, MultinomialNaiveBayes, vectorize, extract_lexical_features
from scipy.sparse import hstack, csr_matrix
from models import compute_alpha_values


nltk.download('stopwords') # comment out after first run
nltk.download('punkt_tab') # comment out after first run
# Get English stopwords
stop_words = {'i', 'my', 'we', 'us', 'not', 'never', 'is', 'are', 'was', 'were', 'could', 'would', 'might', 'should', 'hotel'}


def parse_to_pandas(data_dir: str):
    filenames = []
    dataset = []
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"for path: {data_dir}")
    for polarity in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir,polarity)):
            for truthfulness in os.listdir(os.path.join(data_dir,polarity)):
                for fold in os.listdir(os.path.join(data_dir,polarity,truthfulness)):
                    for file in os.listdir(os.path.join(data_dir,polarity,truthfulness,fold)):
                        filenames.append(os.path.join(data_dir,polarity,truthfulness,fold,file))
    for file in filenames:
        row = {}
        row['dataset'] = file.split('/')[0]
        row['polarity'] = file.split('/')[1]
        row['truthfulness'] = file.split('/')[2]
        row['fold'] = file.split('/')[3]
        row['txt_path'] = file
        dataset.append(row)
    return pd.DataFrame(dataset)

def remove_puncuation(text: str):
    clean_text = "".join([char for char in text if char not in string.punctuation])
    return clean_text

def remove_nums(text: str):
    return "".join([i for i in text if not i.isdigit()])

def remove_stopwords(text: str):
    # Remove extent white spaces
    text = re.sub(' +', ' ', text)
    tokens = word_tokenize(text.lower())
    # Remove stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)
    
def main():
    # the code assumes the txt files are inside a folder named 'op_spam_v1.4' in the project folder
    # 1. preparation
    # df = parse_to_pandas('op_spam_v1.4')
    # df.to_csv('dataset_df.csv')
    
    # 2. pre processing
    df = pd.read_csv('dataset_df.csv')
    # combined_matrix = hstack([vectors, lexical_sparse])
    # all_feature_names = list(vectors_names) + lexical_feature_names
    # 2.1 filter
    df = df[df['polarity'].str.contains('negative')]
    df['text'] = df['txt_path'].map(lambda path: open(path,'r').read())
    
    # 2.2 extract lexical features and normalise them
    logRegScaler = StandardScaler()
    multiNBScaler = MinMaxScaler()
    lexical_df = extract_lexical_features(df.text.tolist())
    lexical_features_logReg = logRegScaler.fit_transform(lexical_df)
    lexical_features_multiNB = multiNBScaler.fit_transform(lexical_df)
    lexical_feature_names = lexical_df.columns.tolist()
    
    # text preprocessing
    df['text_processed'] = df['text'].map(lambda text: remove_puncuation(text))
    df['text_processed'] = df['text_processed'].map(lambda text: text.lower())
    df['text_processed'] = df['text_processed'].map(lambda text: remove_nums(text))
    df['text_processed'] = df['text_processed'].map(lambda text: remove_stopwords(text))
    
    idx = df.index.tolist()
    np.random.shuffle(idx)
    df = df.loc[idx].reset_index(drop=True)
    
    test_fold = 'fold5' 
    train_indexes = df[df['fold'] != test_fold]['index'].tolist()
    test_indexes = df[df['fold'] == test_fold]['index'].tolist()
    
    # 2.5 text vectorization
    # remove words that appear in less thn 5% of the dataset
    for k in [5,10]:
        for maxF in [150,200,250,300,350,400,450,500,550]:
            use_lexical_features = True
            dtm_unigrams, feature_names_unigrams = vectorize(df, max_features=maxF)
            dtm_bigrams, feature_names_bigrams = vectorize(df, with_bigrams=True, max_features=maxF)
            if use_lexical_features:
                dtm_unigrams_logReg = np.hstack((dtm_unigrams, lexical_features_logReg))
                dtm_bigrams_logReg = np.hstack((dtm_bigrams, lexical_features_logReg))
                dtm_unigrams_multiNB = np.hstack((dtm_unigrams, lexical_features_multiNB))
                dtm_bigrams_multiNB = np.hstack((dtm_bigrams, lexical_features_multiNB))
            
            # 3. labeling 0: deceptive, 1: truthful
            labels = np.array(df['txt_path'].map(lambda path: np.array(0) if 'deceptive' in path else np.array(1)).tolist())
            
            # 4. split
            unigrams_ds_logReg = {'X_train': dtm_unigrams_logReg[train_indexes, :], 'X_test': dtm_unigrams_logReg[test_indexes, :]}
            bigrams_ds_logReg = {'X_train': dtm_bigrams_logReg[train_indexes, :], 'X_test': dtm_bigrams_logReg[test_indexes, :]}
            unigrams_ds_multiNB = {'X_train': dtm_unigrams_multiNB[train_indexes, :], 'X_test': dtm_unigrams_multiNB[test_indexes, :]}
            bigrams_ds_multiNB = {'X_train': dtm_bigrams_multiNB[train_indexes, :], 'X_test': dtm_bigrams_multiNB[test_indexes, :]}
            y_train, y_test = labels[train_indexes], labels[test_indexes]
    
            # Run Logistic Regression CV
            LogRegCV(**unigrams_ds_logReg,
                    y_train=y_train , y_test=y_test,
                    k_folds=k, test_fold=test_fold)
            LogRegCV(**bigrams_ds_logReg,
                    y_train=y_train , y_test=y_test, with_bigrams=True,
                    k_folds=k, test_fold=test_fold)
            
            # Run Multinomial NB
            unigrams_alpha_values = compute_alpha_values(X_bow=dtm_unigrams, lexical_X=lexical_features_multiNB, y=labels)
            bigrams_alpha_values = compute_alpha_values(X_bow=dtm_bigrams, lexical_X=lexical_features_multiNB, y=labels)
            MultinomialNaiveBayes(**unigrams_ds_multiNB,
                                y_train=y_train , y_test=y_test,
                                k_folds=k, test_fold=test_fold, alpha=unigrams_alpha_values)
            MultinomialNaiveBayes(**bigrams_ds_multiNB,
                                y_train=y_train , y_test=y_test,
                                with_bigrams=True, k_folds=k, test_fold=test_fold, alpha=bigrams_alpha_values)
            
if __name__ == '__main__':
    main()