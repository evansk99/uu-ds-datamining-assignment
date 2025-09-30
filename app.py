import os
import pandas as pd
import string
import nltk
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle

nltk.download('stopwords') # comment out after first run
nltk.download('punkt_tab') # comment out after first run
# Get English stopwords
stop_words = set(stopwords.words('english'))


def LogReg(X: np.array, y: np.array, with_bigrams=False):
    if with_bigrams:
        return
    
    model = LogisticRegression(random_state=0)
    classifier = model.fit(X,y)
    preds = classifier.predict(X)
    probs = classifier.predict_proba(X)
    logits = classifier.predict_log_proba(X)
    raise ValueError


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

def remove_stopwords(text: str):
    # remove numbers
    for n in ['0','1','2','3','4','5','6','7','8','9']:
        text = text.replace(n, '') if n in text else text
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
    df['text_processed_tokens'] = df['text_processed'].map(lambda text: remove_stopwords(text))
    # shuffle
    df = shuffle(df, random_state=42)
    
    # 2.5 text vectorization
    train_indexes = df[df['fold'] != 'fold5']['index'].tolist()
    test_indexes = df[df['fold'] == 'fold5']['index'].tolist()
    # remove words that appear in less thn 5% of the dataset
    vect = CountVectorizer(min_df=.05)
    vects = vect.fit_transform(df.text_processed_tokens)
    td = pd.DataFrame(vects.todense())
    td.columns = vect.get_feature_names_out()
    td['doc'] = df['txt_path'].map(lambda p: p.split('/')[-1])
    # convert to matrix
    dtm = td.to_numpy()[:, :-1]
    
    # 3. labeling 0: deceptive, 1: truthful
    labels = np.array(df['txt_path'].map(lambda path: np.array(0) if 'deceptive' in path else np.array(1)).tolist())
    
    # 4. split 
    X_train, X_test = dtm[train_indexes, :], dtm[test_indexes, :]
    y_train, y_test = labels[train_indexes], labels[test_indexes]
    
    # Run Model
    LogReg(X_train, y_train)
    
    
if __name__ == '__main__':
    main()