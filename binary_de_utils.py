#       Library with Appearance Differential Expression analysis functions
#
#   Author:     Edward Vitkin
#   email:      edward.vitkin@gmail.com
#   version:    14 Dec, 2023
#
import numpy as np
from scipy import stats

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
            stat[gi], pval[gi] = znormal_proportion_binde_test(pos_vals, neg_vals)
        elif testnm == 'hg':
            if dirsign[gi]>0:
                stat[gi], pval[gi] = None, hg_binde_test_vec(pos_vals, neg_vals)
            else:
                stat[gi], pval[gi] = None, hg_binde_test_vec(neg_vals, pos_vals)

    return stat, dirsign, pval, pcnts, ncnts

#   Hypergeometric distribution based Test on data statistics:
#   INPUTS:
#       Na - number of experiments from type A
#       cA - number of experiments from type A, where the molecule of interest was observed
#       Nb - number of experiments from type B
#       cB - number of experiments from type B, where the molecule of interest was observed
def hg_binde_test(Na, cA, Nb, cB):
    population_sz = Na + Nb
    groupA_sz = Na
    groupB_sz = cA + cB
    intrsct_sz = cA
    pvalue = stats.hypergeom.sf(intrsct_sz-1, population_sz, groupA_sz, groupB_sz)

    return pvalue

#   Hypergeometric distribution based Test on experiment binary indicator vectors
#   INPUTS:
#       grpA - binary appearance indicator of molecule in Group A experiments
#       grpB - binary appearance indicator of molecule in Group B experiments
#           values:
#               0 (False) - molecule was NOT observed in the experiment
#               1 (True) - molecule was observed in the experiment
def hg_binde_test_vec(grpA, grpB):
    Na, Nb = len(grpA), len(grpB)
    cA, cB = np.count_nonzero(grpA), np.count_nonzero(grpB)
    return hg_binde_test(Na, cA, Nb, cB)


#   Proportion based Test on experiment binary indicator vectors
#   INPUTS:
#       grpA - binary appearance indicator of molecule in Group A experiments
#       grpB - binary appearance indicator of molecule in Group B experiments
#           values:
#               0 (False) - molecule was NOT observed in the experiment
#               1 (True) - molecule was observed in the experiment
#       alternative - tested alternatives:
#           two-sided [DEFAULT] - Checks if proportion of appearances of Group A is DIFFERENT of that of Group B
#           less - Checks if proportion of appearances of Group A is LOWER of that of Group B
#           greater [DEFAULT] - Checks if proportion of appearances of Group A is HIGHER of that of Group B
def znormal_proportion_binde_test(grpA, grpB, alternative='two-sided'):
    nA = grpA.shape[0]
    nB = grpB.shape[0]
    hitA = np.count_nonzero(grpA,axis=0)
    hitB = np.count_nonzero(grpB,axis=0)
    pA = hitA / nA
    pB = hitB / nB

    zscore, pvalue = np.zeros(shape=pA.shape), np.ones(shape=pA.shape)

    if np.all((pA+pB)<=0) or np.all((pA+pB)>=2):        #   all-zero OR all-one probabilities
        return zscore, pvalue

    p = (hitA+hitB) / (nA+nB)
    q = 1 - p
    mu__pApB = 0
    sigma_pApB = np.sqrt( p*q/nA + p*q/nB )
    non_zero_sigma = sigma_pApB != 0

    data_mu_pApB = pA - pB

    zscore[non_zero_sigma] = ( data_mu_pApB[non_zero_sigma] - mu__pApB) / sigma_pApB[non_zero_sigma]

    if alternative=='two-sided':
        pvalue[non_zero_sigma] = 2*stats.norm.sf(abs(zscore[non_zero_sigma]))
    elif alternative=='less':
        pvalue[non_zero_sigma] = stats.norm.sf(-zscore[non_zero_sigma])
    elif alternative=='greater':
        pvalue[non_zero_sigma] = stats.norm.sf(zscore[non_zero_sigma])

    return zscore, pvalue

def usage_example():
    grp_1 = np.zeros(shape=(50))
    grp_1[:32] = 1
    grp_2 = np.zeros(shape=(60))
    grp_2[:18] = 1

    ttl_str = lambda grp_1,grp_2: f"({np.count_nonzero(grp_1)}/{len(grp_1)} vs {np.count_nonzero(grp_2)}/{len(grp_2)} successes)"

    print('Proportion-based test (zscore,pvalue)')
    print(f"\t{ttl_str(grp_1, grp_2)}, alternative=two-sided = {znormal_proportion_binde_test(grp_1, grp_2)}")
    print(f"\t{ttl_str(grp_2, grp_1)}, alternative=two-sided = {znormal_proportion_binde_test(grp_2, grp_1)}")
    print(f"\t{ttl_str(grp_1, grp_2)}, alternative=less = {znormal_proportion_binde_test(grp_1, grp_2, 'less')}")
    print(f"\t{ttl_str(grp_1, grp_2)}, alternative=greater = {znormal_proportion_binde_test(grp_1, grp_2, 'greater')}")

    print('HG-based test (pvalue)')
    print(f"\t{ttl_str(grp_1, grp_2)} = {hg_binde_test_vec(grp_1, grp_2)}")
    print(f"\t{ttl_str(grp_2, grp_1)} = {hg_binde_test_vec(grp_2, grp_1)}")
    return


if __name__ == '__main__':
    usage_example()
