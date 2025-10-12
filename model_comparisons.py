import pandas as pd
import os
from scipy.stats import wilcoxon


def model_comparison(
    model1_name: str,
    model1_score: list[int],
    model2_name: str,
    model2_score: list[int],
):
    if len(model1_score) != len(model2_score):
        return NameError("Models need to have been trained on same number of folds")
    
    stat, p = wilcoxon(model1_score, model2_score)
    comps_file = 'wilcoxon_comparisons.csv'
    res = pd.DataFrame([{
        "model1": model1_name,
        "model2": model2_name,
        "stat": stat,
        "p": p
    }])
    if not os.path.exists(comps_file):
        res.to_csv(comps_file, index=False)
    else:
        res.to_csv(comps_file, index=False, header=False, mode='a')


k = int(input('K [5,10,16,32]:'))
n_gram_range = int(input('n_gram_range [1,2]:'))

# Compare MultiNB and LogRegr
with_bigrams = True if n_gram_range == 2 else False
multiNB_df = pd.read_csv('plots/multinomialNB-accuracies-v2.csv')
multiNB_df = multiNB_df[(multiNB_df['k'] == k) & (multiNB_df['with_bigrams'] == with_bigrams)]
bestNB = multiNB_df.loc[multiNB_df['test_accuracy'].idxmax()].to_dict()
bestNB_scores = [float(bestNB[f"fold{key}"]) for key in range(1,k+1)]
logReg_df = pd.read_csv('plots/logRegr-accuracies-v2.csv')
logReg_df = logReg_df[(logReg_df['k'] == k) & (logReg_df['with_bigrams'] == with_bigrams)]
bestLogReg = logReg_df.loc[logReg_df['test_accuracy'].idxmax()].to_dict()
bestLogReg_scores = [float(bestLogReg[f"fold{key}"]) for key in range(1,k+1)]
# model_comparison(f'multiNB_ngram_{n_gram_range}_k_{k}', bestNB_scores, f'logReg_ngram_{n_gram_range}_k_{k}', bestLogReg_scores)

# Compare MultiNB and DecTree
decTree_df = pd.read_csv('dt-accuracies.csv')
decTree = decTree_df[decTree_df['k'] == k].iloc[0].to_dict()
decTree_scores = [float(decTree[f'fold_{key}_acc']) for key in range(1,k+1)]
model_comparison(f'multiNB_ngram_{n_gram_range}_k_{k}', bestNB_scores, f'decTree_k_{k}', decTree_scores)