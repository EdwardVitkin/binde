#       Library with Appearance Differential Expression analysis functions
#
#   Author:     Edward Vitkin
#   email:      edward.vitkin@gmail.com
#   version:    14 Dec, 2023
#
import numpy as np
import pandas as pd
from scipy import stats

import binary_de_utils as binde
#   Calculates appearance probabilities for all molecules in experiments
#   INPUTS:
#       X - measurements of size: [Num Measured molecules]x[Num of experiments]
#       Y - experiment binary indicator vector of size [Num of experiments]
#           1 - experiment belongs to Group A (test)
#           0 - experiment belongs to Group B (control)
#       testnm - type of test to apply
#           'proportion' - Proportion based Test
#           'hg'         - Hypergeometric distribution based Test
#
def calc_app_probability(X,Y, testnm='proportion'):
    print(f'Calculating UQ w <{testnm}>. # Positive={np.count_nonzero(Y == 1)}, # Negative={np.count_nonzero(Y == 0)}')
    stat, pval = np.zeros(shape=X.shape[0]),np.zeros(shape=X.shape[0])
    pcnts, pN = np.sum(np.sign(X[:, Y == 1]), axis=1), np.count_nonzero(Y == 1)
    ncnts, nN = np.sum(np.sign(X[:, Y == 0]), axis=1), np.count_nonzero(Y == 0)
    dirsign = np.sign(pcnts/pN - ncnts/nN)
    for gi in range(X.shape[0]):
        pos_vals = X[gi, Y == 1]
        neg_vals = X[gi, Y == 0]

        if testnm == 'proportion':
            stat[gi], pval[gi] = binde.proportion_binde_test(pos_vals, neg_vals)
        elif testnm == 'hg':
            if dirsign[gi]>0:
                stat[gi], pval[gi] = None, binde.hg_binde_test_vec(pos_vals, neg_vals)
            else:
                stat[gi], pval[gi] = None, binde.hg_binde_test_vec(neg_vals, pos_vals)

    return stat, dirsign, pval, pcnts, ncnts


StudentT_testPaired = lambda row: pd.Series(stats.ttest_rel(row['A'], row['B'], nan_policy='raise'))   #   TTest for paired samples
StudentT_testInd = lambda row: pd.Series(stats.ttest_ind(row['A'], row['B'], nan_policy='raise'))   #   TTest for independent samples
MannWhitneyU_test = lambda row: pd.Series(stats.mannwhitneyu(row['A'], row['B']))   #   Mann-Whitney U test for independent samples
Wilcoxon_test = lambda row: pd.Series(stats.ranksums(row['A'], row['B']))   #   Wilcoxon rank-sum test for independent samples

def analyze_measurement_data(data_dict, deStudentT=False, deMannWhitneyU=False, deWilcoxon=False):
    X, Y, index, POS, NEG = data_dict['X'], data_dict['Y'], data_dict['index'], data_dict['pos_label'], data_dict['neg_label']
    results_df = pd.DataFrame([],index=index)
    ypos = Y==1
    yneg = Y==0

    Npos, Nneg = np.count_nonzero(ypos), np.count_nonzero(yneg)
    results_df['A'] = list(X[:,ypos])
    results_df['B'] = list(X[:,yneg])

    print('------------------ START OF APPEARANCE -----------------------')
    proportion_res = calc_app_probability(X, Y, testnm='proportion')

    results_df['dirsign'] = ['UP' if v>0 else 'DOWN' for v in proportion_res[1]]       #   Expression Direction with respect to BCC. 1== BCC is higher, -1== BCC is lower
    results_df[f'{POS} count'] = proportion_res[3]
    results_df[f'{POS} proportion'] = proportion_res[3] / Npos
    results_df[f'{POS} mean'] = results_df['A'].apply(np.mean)
    results_df[f'{POS} std'] = results_df['A'].apply(np.std)
    results_df[f'{NEG} count'] = proportion_res[4]
    results_df[f'{NEG} proportion'] = proportion_res[4] / Nneg
    results_df[f'{NEG} mean'] = results_df['B'].apply(np.mean)
    results_df[f'{NEG} std'] = results_df['B'].apply(np.std)
    results_df['ratio'] = results_df[f'{POS} mean'] / ( results_df[f'{NEG} mean'] + 1e-8)

    results_df['Proportion_Test STAT'], results_df['Proportion_Test Pvalue'] = proportion_res[0], proportion_res[2]
    results_df['Proportion_Test Rank'] = results_df['Proportion_Test Pvalue'].rank()

    hg_res = calc_app_probability(X, Y, testnm='hg')
    results_df['HG_Test Pvalue'] = hg_res[2]
    results_df['HG_Test Rank'] = results_df['HG_Test Pvalue'].rank()

    print('------------------ END OF APPEARANCE -----------------------')

    prot_freqs = (results_df[f'{POS} count']+results_df[f'{NEG} count']) / float(len(ypos))
    for perc_c in [0.9, 0.5, 0.1]:
        n_of_prots = np.count_nonzero(prot_freqs<perc_c)
        perc_of_samples = n_of_prots / len(prot_freqs)
        print(f" {n_of_prots}, i.e. {100*perc_of_samples:.1f}% of {len(prot_freqs)} proteins appear in less than {perc_c*100:.0f}% of {len(ypos)} samples")

    print('------------------ START OF DE -----------------------')

    if deStudentT:
        results_df[['StudentT STAT', 'StudentT Pvalue']] = results_df.apply(StudentT_test, axis=1)
        results_df['StudentT Rank'] = results_df['StudentT Pvalue'].rank()
    if deMannWhitneyU:
        results_df[['MannWhitneyU STAT', 'MannWhitneyU Pvalue']] = results_df.apply(MannWhitneyU_test, axis=1)
        results_df['MannWhitneyU Rank'] = results_df['MannWhitneyU Pvalue'].rank()
    if deWilcoxon:
        results_df[['Wilcoxon STAT', 'Wilcoxon Pvalue']] = results_df.apply(Wilcoxon_test, axis=1)
        results_df['Wilcoxon Rank'] = results_df['Wilcoxon Pvalue'].rank()

    print('------------------ END OF DE -----------------------')
    results_df.drop(columns=['A','B'], inplace=True)

    results_df['Minimal Pvalue'] = results_df[[c for c in results_df.columns if c.endswith('Pvalue')]].min(axis=1)
    results_df['Minimal Rank'] = results_df[[c for c in results_df.columns if c.endswith('Rank')]].min(axis=1)
    return results_df
