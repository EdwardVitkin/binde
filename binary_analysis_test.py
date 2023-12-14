#       Real-Data usage example of Appearance Differential Expression analysis functions
#   for article "Differential expression analysis of binary appearance patterns"
#
#   Author:     Edward Vitkin
#   email:      edward.vitkin@gmail.com
#   version:    14 Dec, 2023
#
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

import binary_de_utils as binde

pd.set_option("display.colheader_justify","left")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)

FL_PREFIX = 'de_vs_binary'

TOP = 6
showUqFig = True

if __name__ == '__main__':
    pulse_data = pd.read_pickle(FL_PREFIX + '.input.pkl')
    app_results_df = pulse_data['protein_info_df']
    X, Y = pulse_data['X'], pulse_data['Y']
    ypos = Y==1
    yneg = Y==0

    Npos, Nneg = np.count_nonzero(ypos), np.count_nonzero(yneg)
    app_results_df['A'] = list(X[:,ypos])
    app_results_df['B'] = list(X[:,yneg])

    print('------------------ START OF APPEARANCE -----------------------')
    proportion_res = binde.calc_app_probability(X, Y, testnm='proportion')

    app_results_df['dirsign'] = ['UP' if v>0 else 'DOWN' for v in proportion_res[1]]       #   Expression Direction with respect to BCC. 1== BCC is higher, -1== BCC is lower
    app_results_df['BCC count'] = proportion_res[3]
    app_results_df['BCC proportion'] = proportion_res[3] / Npos
    app_results_df['BCC mean'] = app_results_df['A'].apply(np.mean)
    app_results_df['BCC std'] = app_results_df['A'].apply(np.std)
    app_results_df['SCC count'] = proportion_res[4]
    app_results_df['SCC proportion'] = proportion_res[4] / Nneg
    app_results_df['SCC mean'] = app_results_df['B'].apply(np.mean)
    app_results_df['SCC std'] = app_results_df['B'].apply(np.std)
    app_results_df['ratio'] = app_results_df['BCC mean'] / ( app_results_df['SCC mean'] + 1e-8)

    app_results_df['Proportion_Test STAT'], app_results_df['Proportion_Test Pvalue'] = proportion_res[0], proportion_res[2]
    hg_res = binde.calc_app_probability(X, Y, testnm='hg')
    app_results_df['HG_Test Pvalue'] = hg_res[2]

    print('------------------ END OF APPEARANCE -----------------------')

    print('------------------ START OF DE -----------------------')

    StudentT_test = lambda row: pd.Series(stats.ttest_ind(row['A'], row['B'], nan_policy='raise'))   #   TTest for independent samples
    MannWhitneyU_test = lambda row: pd.Series(stats.mannwhitneyu(row['A'], row['B']))   #   Mann-Whitney U test for independent samples
    Wilcoxon_test = lambda row: pd.Series(stats.ranksums(row['A'], row['B']))   #   Wilcoxon rank-sum test

    # app_results_df[['StudentT STAT', 'StudentT Pvalue']] = app_results_df.apply(StudentT_test, axis=1)
    # app_results_df[['MannWhitneyU STAT', 'MannWhitneyU Pvalue']] = app_results_df.apply(MannWhitneyU_test, axis=1)
    app_results_df[['Wilcoxon STAT', 'Wilcoxon Pvalue']] = app_results_df.apply(Wilcoxon_test, axis=1)

    app_results_df.drop(columns=['A','B'], inplace=True)
    print('------------------ END OF DE -----------------------')

    app_results_ranks_df = app_results_df[['Protein ID', 'Protein names', 'Gene names', 'dirsign',
                                           'BCC count',  'BCC proportion',  'SCC count',  'SCC proportion'
                                           ]].copy()
    app_results_ranks_df['Proportion_Test Rank'] = app_results_df['Proportion_Test Pvalue'].rank()
    app_results_ranks_df['HG_Test Rank'] = app_results_df['HG_Test Pvalue'].rank()
    # app_results_ranks_df['StudentT Rank'] = app_results_df['StudentT Pvalue'].rank()
    # app_results_ranks_df['MannWhitneyU Rank'] = app_results_df['MannWhitneyU Pvalue'].rank()
    app_results_ranks_df['Wilcoxon Rank'] = app_results_df['Wilcoxon Pvalue'].rank()

    app_results_ranks_df['Minimal Rank'] = app_results_ranks_df[[c for c in app_results_ranks_df.columns if c.endswith('Rank')]].min(axis=1)

    main_out_df = app_results_ranks_df[app_results_ranks_df['Minimal Rank']<=TOP].sort_values('Minimal Rank')
    main_out_df = main_out_df.join(app_results_df[['Proportion_Test Pvalue', 'HG_Test Pvalue','Wilcoxon Pvalue',
                                                   'BCC mean', 'SCC mean','BCC std', 'SCC std', 'ratio']])
    main_out_df.set_index('Gene names', inplace=True)
    main_out_df = main_out_df[['Protein names','dirsign','ratio',
                               'BCC count',  'BCC proportion',  'BCC mean','BCC std',
                               'SCC count',  'SCC proportion', 'SCC mean','SCC std',
                              'Wilcoxon Pvalue',  'Wilcoxon Rank',
                              'Proportion_Test Pvalue',  'Proportion_Test Rank',
                              'HG_Test Pvalue',  'HG_Test Rank',
                                           ]]
    print(main_out_df)

    main_out_df.to_csv(FL_PREFIX + '.main.csv')
    app_results_df.to_csv(FL_PREFIX + '.pvalues.csv', index=False)
    app_results_ranks_df.to_csv(FL_PREFIX + '.ranks.csv', index=False)

    pv_Pr,pv_Sr = app_results_df.corr(numeric_only=True), app_results_df.corr('spearman',numeric_only=True)
    rnk_Pr,rnk_Sr = app_results_ranks_df.corr(numeric_only=True), app_results_ranks_df.corr('spearman',numeric_only=True)
    atT = lambda df,c1,c2: df.at[c1,c2]
    print('Proportion vs HG:')
    print(f"\tPearson: Rank={atT(rnk_Pr,'Proportion_Test Rank','HG_Test Rank'):.2f}, Pvalue={atT(pv_Pr,'Proportion_Test Pvalue','HG_Test Pvalue'):.2f}")
    print(f"\tSpearman: Rank={atT(rnk_Sr,'Proportion_Test Rank','HG_Test Rank'):.2f}, Pvalue={atT(pv_Sr,'Proportion_Test Pvalue','HG_Test Pvalue'):.2f}")
    print('Proportion vs Wilcoxon:')
    print(f"\tPearson: Rank={atT(rnk_Pr,'Proportion_Test Rank','Wilcoxon Rank'):.2f}, Pvalue={atT(pv_Pr,'Proportion_Test Pvalue','Wilcoxon Pvalue'):.2f}")
    print(f"\tSpearman: Rank={atT(rnk_Sr,'Proportion_Test Rank','Wilcoxon Rank'):.2f}, Pvalue={atT(pv_Sr,'Proportion_Test Pvalue','Wilcoxon Pvalue'):.2f}")
    print('HG vs Wilcoxon:')
    print(f"\tPearson: Rank={atT(rnk_Pr,'HG_Test Rank','Wilcoxon Rank'):.2f}, Pvalue={atT(pv_Pr,'HG_Test Pvalue','Wilcoxon Pvalue'):.2f}")
    print(f"\tSpearman: Rank={atT(rnk_Sr,'HG_Test Rank','Wilcoxon Rank'):.2f}, Pvalue={atT(pv_Sr,'HG_Test Pvalue','Wilcoxon Pvalue'):.2f}")

    if showUqFig:
        get_uq = lambda df, zlbl, elbl: df[ (df[zlbl] ==0) & (df[elbl] >0)][[elbl]+[c for c in df.columns if (c.endswith('Rank')|c.endswith('Pvalue'))]].groupby(elbl).mean().sort_index(ascending=False)
        uq_BCC = get_uq(app_results_ranks_df, 'SCC proportion','BCC proportion').join(get_uq(app_results_df, 'SCC proportion','BCC proportion'))
        uq_SCC = get_uq(app_results_ranks_df, 'BCC proportion','SCC proportion').join(get_uq(app_results_df, 'BCC proportion','SCC proportion'))
        uq_BCC.index, uq_SCC.index = uq_BCC.index*100, uq_SCC.index*100

        plt.subplot(2,2,1)
        plt.plot(uq_BCC['Proportion_Test Rank'])
        plt.plot(uq_BCC['HG_Test Rank'])
        plt.yscale("log");plt.yticks(fontsize=12);plt.xticks(fontsize=12);
        plt.legend(['Proportion_Test', 'HG_Test'], fontsize=14)
        plt.ylabel('Log10 of gene Rank', fontsize=16)
        plt.title('Genes unique to BCC', fontsize=20)
        plt.subplot(2,2,3)
        plt.plot(uq_BCC['Proportion_Test Pvalue'])
        plt.plot(uq_BCC['HG_Test Pvalue'])
        plt.yscale("log");plt.yticks(fontsize=12);plt.xticks(fontsize=12);
        plt.legend(['Proportion_Test', 'HG_Test'], fontsize=14)
        plt.ylabel('Log10 of gene Pvalue', fontsize=16)
        plt.xlabel('Proportion within BCC samples [%]', fontsize=16)

        plt.subplot(2,2,2)
        plt.plot(uq_SCC['Proportion_Test Rank'])
        plt.plot(uq_SCC['HG_Test Rank'])
        plt.yscale("log");plt.yticks(fontsize=12);plt.xticks(fontsize=12);
        plt.legend(['Proportion_Test', 'HG_Test'], fontsize=14)
        plt.ylabel('Log10 of gene Rank', fontsize=16)
        plt.title('Genes unique to SCC', fontsize=20)
        plt.subplot(2,2,4)
        plt.plot(uq_SCC['Proportion_Test Pvalue'])
        plt.plot(uq_SCC['HG_Test Pvalue'])
        plt.yscale("log");plt.yticks(fontsize=12);plt.xticks(fontsize=12);
        plt.legend(['Proportion_Test', 'HG_Test'], fontsize=14)
        plt.ylabel('Log10 of gene Pvalue', fontsize=16)
        plt.xlabel('Proportion within SCC samples [%]', fontsize=16)
        plt.show()
    print('Finished')