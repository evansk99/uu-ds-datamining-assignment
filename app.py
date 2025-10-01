import os
import pandas as pd
import string
import nltk
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
from models import LogRegCV, MultinomialNaiveBayes, vectorize

nltk.download('stopwords') # comment out after first run
nltk.download('punkt_tab') # comment out after first run
# Get English stopwords
stop_words = set(stopwords.words('english'))


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
    # 2.1 filter
    df = df[df['polarity'].str.contains('negative')]
    df['text'] = df['txt_path'].map(lambda path: open(path,'r').read())
    df['text_processed'] = df['text'].map(lambda text: remove_puncuation(text))
    df['text_processed'] = df['text_processed'].map(lambda text: text.lower())
    df['text_processed'] = df['text_processed'].map(lambda text: remove_nums(text))
    df['text_processed'] = df['text_processed'].map(lambda text: remove_stopwords(text))
    train_indexes = df[df['fold'] != 'fold5']['index'].tolist()
    test_indexes = df[df['fold'] == 'fold5']['index'].tolist()
    
    # 2.5 text vectorization
    # remove words that appear in less thn 5% of the dataset
    dtm_unigrams = vectorize(df)
    dtm_bigrams = vectorize(df, with_bigrams=True)
    
    # 3. labeling 0: deceptive, 1: truthful
    labels = np.array(df['txt_path'].map(lambda path: np.array(0) if 'deceptive' in path else np.array(1)).tolist())
    
    # 4. split
    unigrams_ds = {'X_train': dtm_unigrams[train_indexes, :], 'X_test': dtm_unigrams[test_indexes, :]}
    bigrams_ds = {'X_train': dtm_bigrams[train_indexes, :], 'X_test': dtm_bigrams[test_indexes, :]}
    y_train, y_test = labels[train_indexes], labels[test_indexes]
    
    # Run Logistic Regression CV
    LogRegCV(**unigrams_ds,y_train=y_train , y_test=y_test)
    LogRegCV(**bigrams_ds,y_train=y_train , y_test=y_test, with_bigrams=True)
    
    # Run Multinomial NB
    MultinomialNaiveBayes(**unigrams_ds,y_train=y_train , y_test=y_test)
    MultinomialNaiveBayes(**bigrams_ds,y_train=y_train , y_test=y_test, with_bigrams=True)
    
if __name__ == '__main__':
    main()