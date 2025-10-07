import os
import pandas as pd
import string
import nltk
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer  ## Me
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
from models import LogRegCV, MultinomialNaiveBayes, run_tree_experiment,tune_and_evaluate, vectorize #Me
from sklearn.feature_extraction.text import TfidfVectorizer # test it

#nltk.download('stopwords') # comment out after first run
#nltk.download('punkt_tab') # comment out after first run

from collections import Counter ##


# Get English stopwords
stop_words = set(stopwords.words('english'))
print(stop_words)
#stop_words -= {"not", "very"} #Included these two.
stop_words -={'i', 'my', 'we', 'us', 'not', 'never', 'is', 'are', 'was', 'were', 'could', 'would', 'might', 'should', 'very'}
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
    # 1. Preparation
    df = parse_to_pandas('Data') #Changed op_spam to Data
    df.to_csv('dataset_df.csv')
    
    # 2. Pre-processing
    df = pd.read_csv('dataset_df.csv')

    df = df[df['polarity'].str.contains('negative')]
    df['text'] = df['txt_path'].map(lambda path: open(path,'r').read())
    df['text_processed'] = df['text'].map(lambda text: remove_puncuation(text))
    df['text_processed'] = df['text_processed'].map(lambda text: text.lower())
    df['text_processed'] = df['text_processed'].map(lambda text: remove_nums(text))
    df['text_processed'] = df['text_processed'].map(lambda text: remove_stopwords(text)) 
    
    #Unique Words Before Removing  Stop Words
    text_deceipt = "".join(df.iloc[0:400]['text_processed'])
    words_deceipt = text_deceipt.split()
    word_count0 = Counter(words_deceipt)
    unique_dec = len(set(words_deceipt))
    top_10_dec = word_count0.most_common(30)


    text_true = "".join(df.iloc[400:800]['text_processed'])
    words_true =  text_true.split()
    word_count1 = Counter(words_true)
    unique_true = len(set(words_true))
    top_10_true = word_count1.most_common(30)
    
    print("Total Words 0: ",len(words_deceipt))
    print("Unique Words 0: ", unique_dec)
    print("Top 10 Words 0: ", top_10_dec)
    print("Total Words 1: ", len(words_true))
    print("Unique Words 1: ", unique_true)
    print("Top 10 Words 1: ", top_10_true)
    
    df_common_words=pd.DataFrame({"Top 10 Dec": top_10_dec, "Top 10 True" : top_10_true})
    print(df_common_words)
    
    
    import matplotlib.pyplot as plt
    
    # Example: top words and counts (replace with your actual data)
    top_10_dec = [('i', 2232), ('was', 1936), ('room', 969), ('we', 905), ('hotel', 891),
                  ('my', 815), ('not', 751), ('were', 521), ('chicago', 421), ('is', 359)]
    
    top_10_true = [('was', 1523), ('i', 1445), ('room', 833), ('we', 886), ('hotel', 719),
                   ('not', 689), ('is', 624), ('were', 504), ('my', 494), ('stay', 272)]
    
    # Convert to DataFrame for easier plotting
    df_dec = pd.DataFrame(top_10_dec, columns=['word','count'])
    df_true = pd.DataFrame(top_10_true, columns=['word','count'])
    
    # Ensure the words are in the same order for side-by-side bars
    words = df_dec['word']
    dec_counts = df_dec['count']
    true_counts = df_true.set_index('word').reindex(words)['count']
    
    # Plot
    x = range(len(words))
    width = 0.35  # bar width
    
    fig, ax = plt.subplots(figsize=(12,6))
    ax.bar([i - width/2 for i in x], dec_counts, width, label='Deceptive', color='salmon')
    ax.bar([i + width/2 for i in x], true_counts, width, label='Truthful', color='skyblue')
    
    ax.set_xticks(x)
    ax.set_xticklabels(words, rotation=45, ha='right')
    ax.set_ylabel('Word Count')
    ax.set_title('Top 10 Most Frequent Words per Class (with stopwords)')
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    #Train and test data positions
    train_indexes = df[df['fold'] != 'fold5'].index.tolist()
    test_indexes = df[df['fold'] == 'fold5'].index.tolist()

    #Extra Features Text length and Sentiment    
    analyzer = SentimentIntensityAnalyzer()
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    df['sentiment'] = df['text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    df['exclam_count'] = df['text'].apply(lambda x: x.count('!'))
    #df['neg_score'] = df['text'].apply(lambda x: analyzer.polarity_scores(x)['neg'])
    
    #Split Dataset
    x_vec_tr=df.loc[train_indexes,'text_processed']
    x_vec_test=df.loc[test_indexes,'text_processed']

    # Labeling 0 : Deceptive, 1 : Truthful
    labels = np.array(df['txt_path'].map(lambda path: np.array(0) if 'deceptive' in path else np.array(1)).tolist())

    # Experiment settings
    
    #For Decision Tree Performance on #Features Uni VS Uni+Bigrams & extra features
    # train_results, test_results, model = run_tree_experiment(
    #    df, train_indexes, test_indexes, labels, model_type='decision_tree',
    #    extra_train_features=df.iloc[train_indexes][['word_count','sentiment']], 
    #    extra_test_features=df.iloc[test_indexes][['word_count','sentiment']],extra_features=False
    # )
    
    # Run Decision Tree hyperparameter tuning
    # train_acc=[]
    # test_acc=[]
    # params=[]
    # num_features = [50, 100, 200, 500, 1000]
    # for num in num_features:
    #    best_dt, dt_params, dt_train_score, dt_acc, dt_f1 = tune_and_evaluate(
    #   x_vec_tr, x_vec_test, labels, train_indexes, test_indexes,
    #    model_type="dt", n_features=num, ngram_range=(1,1)) # unigrams+bigrams
    #    train_acc.append(dt_train_score)
    #    test_acc.append(dt_acc)
    #    params.append(dt_params)
    # results_df = pd.DataFrame({
    #    'num_features': num_features,
    #    'train_acc': train_acc,
    #    'test_acc': test_acc
    #    #'params': params
    # })
    
    #print(train_results)
    #print(test_results)  
    
    # Random Forest Performance on #Features +extra features
    train_acc=[]
    test_acc=[]
    params=[]
    train_results, test_results, model = run_tree_experiment(
      df, train_indexes, test_indexes, labels, model_type='random_forest',
      extra_train_features=df.iloc[train_indexes][['word_count','sentiment','exclam_count']], 
      extra_test_features=df.iloc[test_indexes][['word_count','sentiment','exclam_count']],extra_features=True
    )
    
    print(train_results)
    print(test_results) 
    
    # # Run Random Forest hyperparameter tuning
    num_features = [500, 1000, 2000]
    for num in num_features:
        best_rf, rf_params, rf_train_score, rf_acc, rf_f1 = tune_and_evaluate(
        x_vec_tr, x_vec_test, labels, train_indexes, test_indexes,
        model_type="rf", n_features=num, ngram_range=(1,2))
        train_acc.append(rf_train_score)
        test_acc.append(rf_acc)
        params.append(rf_params)
    
    results_df = pd.DataFrame({
    'num_features': num_features,
    'train_acc': train_acc,
    'test_acc': test_acc
    #'params': params
    })
    
    

    # Print nicely
    print(results_df)
    #model.feature_importances_
    
if __name__ == '__main__':
    main()