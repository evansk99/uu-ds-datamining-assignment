import pandas as pd
import os
from scipy.stats import wilcoxon

NB_SCORES_PATH = 'plots/multinomialNB-accuracies-v2.csv'
LOGREG_SCORES_PATH = 'plots/logRegr-accuracies-v2.csv'
DECTREE_SCORES_PATH = 'dt-accuracies.csv'
RANDFOR_SCORES_PATH = 'rf-accuracies.csv'


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


k = 5
# MultiNB vs LogRegr
multiNB_df = pd.read_csv(NB_SCORES_PATH)
multiNB_df = multiNB_df[(multiNB_df['k'] == k)]
bestNB = multiNB_df.loc[multiNB_df['test_accuracy'].idxmax()].to_dict()
bestNB_scores = [float(bestNB[f"fold{key}"]) for key in range(1,k+1)]
logReg_df = pd.read_csv(LOGREG_SCORES_PATH)
logReg_df = logReg_df[(logReg_df['k'] == k)]
bestLogReg = logReg_df.loc[logReg_df['test_accuracy'].idxmax()].to_dict()
bestLogReg_scores = [float(bestLogReg[f"fold{key}"]) for key in range(1,k+1)]
model_comparison(f'multiNB_k_{k}', bestNB_scores, f'logReg_k_{k}', bestLogReg_scores)

# MultiNB vs DecTree
decTree_df = pd.read_csv(DECTREE_SCORES_PATH)
decTree = decTree_df[decTree_df['k'] == k].iloc[0].to_dict()
decTree_scores = [float(decTree[f'fold_{key}_acc']) for key in range(1,k+1)]
model_comparison(f'multiNB_k_{k}', bestNB_scores, f'decTree_k_{k}', decTree_scores)

# LogReg vs DecTree
model_comparison(f'logReg_k_{k}', bestLogReg_scores, f'decTree_k_{k}', decTree_scores)

# MultiNB vs RandForr
randForr_df = pd.read_csv(RANDFOR_SCORES_PATH)
randForr = randForr_df[randForr_df['k'] == k].iloc[0].to_dict()
randForr_scores = [float(randForr[f'fold_{key}_acc']) for key in range(1,k+1)]
model_comparison(f'multiNB_k_{k}', bestNB_scores, f'randForr_k_{k}', randForr_scores)

# LogReg vs RandForr
model_comparison(f'logReg_k_{k}', bestLogReg_scores, f'randForr_k_{k}', randForr_scores)

# MultiNB vs tree ensemble

# LogReg vs tree ensemble

# DecTree vs tree ensemble

# RandForr vs tree ensemble