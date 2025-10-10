import os
import pandas as pd
import string
import nltk
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from models import LogRegCV, MultinomialNaiveBayes, parameter_search, tune_and_evaluate, vectorize #Me


#nltk.download('stopwords') # comment out after first run
#nltk.download('punkt_tab') # comment out after first run

#My imports
from collections import Counter 
import matplotlib.pyplot as plt

# Get English stopwords
stop_words = set(stopwords.words('english'))

#Added term 'very' to the stopwords.
stop_words -={'i', 'my', 'we', 'us', 'not', 'never', 'is', 'are', 'was', 'were', 'could', 'would', 'might', 'should', 'very'}

#Unchanged
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
        row['dataset'] = file.split('\\')[0]   # Changed \ into \\ to run locally.
        row['polarity'] = file.split('\\')[1]
        row['truthfulness'] = file.split('\\')[2]
        row['fold'] = file.split('\\')[3]
        row['txt_path'] = file
        dataset.append(row)
    return pd.DataFrame(dataset)

#Unchanged
def remove_puncuation(text: str):
    clean_text = "".join([char for char in text if char not in string.punctuation])
    return clean_text

#Unchanged
def remove_nums(text: str):
    return "".join([i for i in text if not i.isdigit()])

#Unchanged
def remove_stopwords(text: str):
    # Remove extent white spaces
    text = re.sub(' +', ' ', text)
    tokens = word_tokenize(text.lower())
    # Remove stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)
    
    
def main():
    # the code assumes the txt files are inside a folder named 'op_spam_v1.4' in the project folder
    
    # 1. Preparation
    df = parse_to_pandas('Data') ##Changed op_spam to Data but irrelevant
    df.to_csv('dataset_df.csv')
    
    # 2. Pre-processing
    df = pd.read_csv('dataset_df.csv')
    df = df[df['polarity'].str.contains('negative')]
    df['text'] = df['txt_path'].map(lambda path: open(path,'r').read())
    df['text_processed'] = df['text'].map(lambda text: remove_puncuation(text))
    df['text_processed'] = df['text_processed'].map(lambda text: text.lower())
    df['text_processed'] = df['text_processed'].map(lambda text: remove_nums(text))
    df['text_processed'] = df['text_processed'].map(lambda text: remove_stopwords(text)) 
    
    ### Feature Extraction ### + exploratory analysis
    #Unique Words Before Removing  Stop Words
    text_deceipt = "".join(df.iloc[0:400]['text_processed']) #Class 0
    words_deceipt = text_deceipt.split()
    word_count0 = Counter(words_deceipt)
    unique_deceipt = len(set(words_deceipt))
    top_10_deceipt = word_count0.most_common(10)


    text_true = "".join(df.iloc[400:800]['text_processed']) #Class 1
    words_true =  text_true.split()
    word_count1 = Counter(words_true)
    unique_true = len(set(words_true))
    top_10_true = word_count1.most_common(30)
    
    # print("Total Words 0: ",len(words_deceipt))
    # print("Unique Words 0: ", unique_deceipt)
    # print("Top 10 Words 0: ", top_10_deceipt)
    # print("Total Words 1: ", len(words_true))
    # print("Unique Words 1: ", unique_true)
    # print("Top 10 Words 1: ", top_10_true)
    
    # df_common_words=pd.DataFrame({"Top 10 Dec": top_10_deceipt, "Top 10 True" : top_10_true})
    # print(df_common_words)

    ### Feature Extraction ###
    #Train and test data positions
    train_indexes = df[df['fold'] != 'fold5'].index.tolist()
    test_indexes = df[df['fold'] == 'fold5'].index.tolist()

    #Extra Features Text length and Sentiment (include as extra features)
    analyzer = SentimentIntensityAnalyzer()
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    df['sentiment'] = df['text'].apply(lambda x: analyzer.polarity_scores(x)['compound']) #overal negativity
    df['exclam_count'] = df['text'].apply(lambda x: x.count('!')) #count exclamations -->exclude if you want
    
    #Split Dataset (defined seperately)
    x_train=df.loc[train_indexes,'text_processed']
    x_test=df.loc[test_indexes,'text_processed']

    # Labeling 0 : Deceptive, 1 : Truthful 
    labels = np.array(df['txt_path'].map(lambda path: np.array(0) if 'deceptive' in path else np.array(1)).tolist())

    ### Experiment Setup ##
    
    ### Choose Basic Parameters ####features, unigrams vs bigrams, vectorizer
    results_df = parameter_search( ## name it as you wish haha
        df=df,
        train_indexes=train_indexes,
        test_indexes=test_indexes,
        labels=labels,
        model_type='decision_tree', #or random_forest
        vectorizer_types=['count', 'tfidf', 'tfidf_noidf'],
        feature_sizes=[100, 200, 500, 1000, 2000],
        ngram_options=[1, 2],
        extra_features=False,
        output_excel="Model_Comparisons.xlsx"
    )
    
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', 120)
    # print(results_df)
    
    
    ###Find Best Model Through GridSearch (exhastive search)
    best_model, perf_df, top_features = tune_and_evaluate(
        x_train, #unecessary, could put 
        x_test,
        labels,
        train_indexes,
        test_indexes,
        model_type="dt",
        n_features=200,
        ngram_range=(1,1),
        extra_features=False
    )    
            
    #best rf without extra features (11FN 13FP) --> 85% acc
    #model_type="rf",
    #n_features=2000,
    #ngram_range=(1,2)
    
    #best dt without extra features --> 70% acc
    #model_type="dt",
    #n_features=200,
    #ngram_range=(1,1)

    
if __name__ == '__main__':
    main()
