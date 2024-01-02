#       Transcriptomic-Data usage example of Appearance Differential Expression analysis functions
#   for article "Differential expression analysis of binary appearance patterns"
#
#   Author:     Edward Vitkin
#   email:      edward.vitkin@gmail.com
#   version:    18 Dec, 2023
#
import numpy as np
import pandas as pd
from scipy import stats

import binary_de_utils as binde
import analysis_utils as au

pd.set_option("display.colheader_justify","left")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)
pd.set_option('mode.chained_assignment', None)

BASE_FL_PREFIX = 'transcriptomics/transcriptomics.de_vs_binary'

TOP = 5
showUqFig = False
PVAL_CUT = 0.05

def fdr_str(observed, total, cut=PVAL_CUT):
    expected = total * cut
    fdr = expected/(observed+1e-8)
    s = f"Found {observed}/{total} genes w Pvalue below {cut} (FDR={fdr:.2f})"
    return s


def compare_conditions(rna_data, condAnm,condBnm, NOISE_CUT=np.log2(2)):
    print(f'Comparing conditions: {condAnm} vs {condBnm}. NOISE_CUT={NOISE_CUT}')
    condA = rna_data[f'is_{condAnm}']
    condB = rna_data[f'is_{condBnm}']
    FL_PREFIX = f'{BASE_FL_PREFIX}.{condAnm}vs{condBnm}'
    app_results_df = rna_data['gene_annot'][['Symbol','Description']]

    X = rna_data['data'].values[:,condA|condB]
    print(f'Dataset shape: {X.shape}. Significant corrected pvalue corresponding to {PVAL_CUT} is  {PVAL_CUT/X.shape[0]}')
    Y = condA[condA|condB]
    ypos = Y==1
    yneg = Y==0

    X[X<NOISE_CUT] = 0

    Npos, Nneg = np.count_nonzero(ypos), np.count_nonzero(yneg)
    app_results_df['A'] = list(X[:,ypos])
    app_results_df['B'] = list(X[:,yneg])

    print('------------------ START OF APPEARANCE -----------------------')
    proportion_res = au.calc_app_probability(X, Y, testnm='proportion')

    app_results_df['dirsign'] = ['UP' if v>0 else 'DOWN' for v in proportion_res[1]]       #   Expression Direction with respect to BCC. 1== A is higher, -1== A is lower
    app_results_df[f'{condAnm} count'] = proportion_res[3]
    app_results_df[f'{condAnm} proportion'] = proportion_res[3] / Npos
    app_results_df[f'{condAnm} mean'] = app_results_df['A'].apply(np.mean)
    app_results_df[f'{condAnm} std'] = app_results_df['A'].apply(np.std)
    app_results_df[f'{condBnm} count'] = proportion_res[4]
    app_results_df[f'{condBnm} proportion'] = proportion_res[4] / Nneg
    app_results_df[f'{condBnm} mean'] = app_results_df['B'].apply(np.mean)
    app_results_df[f'{condBnm} std'] = app_results_df['B'].apply(np.std)
    app_results_df['ratio'] = 2**(app_results_df[f'{condAnm} mean'] - app_results_df[f'{condBnm} mean'])         #   Log-scale

    prot_freqs = (app_results_df[f'{condAnm} count']+app_results_df[f'{condBnm} count']) / len(ypos)
    for perc_c in [0.9, 0.5, 0.1]:
        n_of_prots = np.count_nonzero(prot_freqs<perc_c)
        perc_of_samples = n_of_prots / len(prot_freqs)
        print(f" {n_of_prots}, i.e. {100*perc_of_samples:.1f}% of {len(prot_freqs)} genes appear in less than {perc_c*100:.0f}% of samples")

    app_results_df['Proportion_Test STAT'], app_results_df['Proportion_Test Pvalue'] = proportion_res[0], proportion_res[2]
    print(f"\tProportion_Test: {fdr_str(observed=np.count_nonzero(app_results_df['Proportion_Test Pvalue']<PVAL_CUT), total=len(app_results_df), cut=PVAL_CUT)}" )

    hg_res = au.calc_app_probability(X, Y, testnm='hg')
    app_results_df['HG_Test Pvalue'] = hg_res[2]
    print(f"\tHG_Test: {fdr_str(observed=np.count_nonzero(app_results_df['HG_Test Pvalue']<PVAL_CUT), total=len(app_results_df), cut=PVAL_CUT)}" )

    print('------------------ END OF APPEARANCE -----------------------')

    print('------------------ START OF DE -----------------------')

    app_results_df[['StudentT STAT', 'StudentT Pvalue']] = app_results_df.apply(au.StudentT_testInd, axis=1)
    print(fdr_str(observed=np.count_nonzero(app_results_df['StudentT Pvalue']<PVAL_CUT), total=len(app_results_df), cut=PVAL_CUT) )

    app_results_df.drop(columns=['A','B'], inplace=True)
    print('------------------ END OF DE -----------------------')

    app_results_ranks_df = app_results_df[['Symbol', 'Description', 'dirsign',
                                           f'{condAnm} count',  f'{condAnm} proportion',  f'{condBnm} count',  f'{condBnm} proportion'
                                           ]].copy()
    app_results_ranks_df['Proportion_Test Rank'] = app_results_df['Proportion_Test Pvalue'].rank(method='min')
    app_results_ranks_df['HG_Test Rank'] = app_results_df['HG_Test Pvalue'].rank(method='min')
    app_results_ranks_df['StudentT Rank'] = app_results_df['StudentT Pvalue'].rank(method='min')

    app_results_ranks_df['Minimal Rank'] = app_results_ranks_df[[c for c in app_results_ranks_df.columns if c.endswith('Rank')]].min(axis=1)

    main_out_df = app_results_ranks_df[app_results_ranks_df['Minimal Rank']<=TOP].sort_values(['Minimal Rank'])
    main_out_df = main_out_df.join(app_results_df[['Proportion_Test Pvalue', 'HG_Test Pvalue','StudentT Pvalue',
                                                   f'{condAnm} mean', f'{condBnm} mean',f'{condAnm} std', f'{condBnm} std', 'ratio']])
    main_out_df.set_index('Symbol', inplace=True)
    main_out_df = main_out_df[['Description','dirsign','ratio',
                               f'{condAnm} count',  f'{condAnm} proportion',  f'{condAnm} mean',f'{condAnm} std',
                               f'{condBnm} count',  f'{condBnm} proportion', f'{condBnm} mean',f'{condBnm} std',
                              'StudentT Pvalue',  'StudentT Rank',
                              'Proportion_Test Pvalue',  'Proportion_Test Rank',
                              'HG_Test Pvalue',  'HG_Test Rank',
                              'Minimal Rank',
                                           ]].sort_values(['Minimal Rank','ratio'])
    print(main_out_df)

    main_out_df.to_csv(FL_PREFIX + '.main.csv')
    app_results_df.to_csv(FL_PREFIX + '.pvalues.csv')
    app_results_ranks_df.to_csv(FL_PREFIX + '.ranks.csv')

    pv_Pr,pv_Sr = app_results_df.corr(), app_results_df.corr('spearman')
    rnk_Pr,rnk_Sr = app_results_ranks_df.corr(), app_results_ranks_df.corr('spearman')
    atT = lambda df,c1,c2: df.at[c1,c2]
    print('Proportion vs HG:')
    print(f"\tPearson: Rank={atT(rnk_Pr,'Proportion_Test Rank','HG_Test Rank'):.2f}, Pvalue={atT(pv_Pr,'Proportion_Test Pvalue','HG_Test Pvalue'):.2f}")
    print(f"\tSpearman: Rank={atT(rnk_Sr,'Proportion_Test Rank','HG_Test Rank'):.2f}, Pvalue={atT(pv_Sr,'Proportion_Test Pvalue','HG_Test Pvalue'):.2f}")
    print('Proportion vs StudentT:')
    print(f"\tPearson: Rank={atT(rnk_Pr,'Proportion_Test Rank','StudentT Rank'):.2f}, Pvalue={atT(pv_Pr,'Proportion_Test Pvalue','StudentT Pvalue'):.2f}")
    print(f"\tSpearman: Rank={atT(rnk_Sr,'Proportion_Test Rank','StudentT Rank'):.2f}, Pvalue={atT(pv_Sr,'Proportion_Test Pvalue','StudentT Pvalue'):.2f}")
    print('HG vs StudentT:')
    print(f"\tPearson: Rank={atT(rnk_Pr,'HG_Test Rank','StudentT Rank'):.2f}, Pvalue={atT(pv_Pr,'HG_Test Pvalue','StudentT Pvalue'):.2f}")
    print(f"\tSpearman: Rank={atT(rnk_Sr,'HG_Test Rank','StudentT Rank'):.2f}, Pvalue={atT(pv_Sr,'HG_Test Pvalue','StudentT Pvalue'):.2f}")

    print('Finished')

if __name__ == '__main__':
    rna_data = pd.read_pickle(BASE_FL_PREFIX + '.input.pkl')

    bad_genes = rna_data['gene_annot']['Description'].str.startswith('uncharacterized LOC')
    bad_genes = bad_genes | rna_data['gene_annot']['Description'].str.contains('non-protein coding RNA')
    bad_genes = bad_genes | rna_data['gene_annot']['Description'].str.contains('open reading frame')
    bad_genes = bad_genes | rna_data['gene_annot']['Symbol'].str.startswith('LOC')
    bad_genes.fillna(True, inplace=True)

    rna_data['gene_annot'] = rna_data['gene_annot'].loc[~bad_genes]
    rna_data['data'] = rna_data['data'].loc[~bad_genes].apply(np.log2).fillna(0)    #   Omit unknown genes and apply Log2 transform
    print(f'Omitting {np.count_nonzero(bad_genes)} non-protein encoding genes')

    compare_conditions(rna_data, 'BCC', 'SCC')
